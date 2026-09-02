import typing as tp

import torch
from ignite import distributed as idist
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import Engine, Events, _prepare_batch
from ignite.metrics import RootMeanSquaredError
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


def setup_metrics(engine: Engine, suffix: str = "") -> tp.List[str]:
    rmse = RootMeanSquaredError()
    rmse.attach(engine, f"rmse{suffix}")
    return [f"rmse{suffix}"]


def attach_evaluator(
    trainer: Engine, evaluator: Engine, data: DataLoader, call_every: int
):
    def eval_fn():
        evaluator.run(data)

        status_string = [f"Epoch: {trainer.state.epoch}"]
        for metric_name, metric_value in evaluator.state.metrics.items():
            status_string.append(f"{metric_name}: {metric_value:.3f}")
        if idist.get_rank() == 0:
            print(" | ".join(status_string))

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        lambda engine: eval_fn() if engine.state.iteration % call_every == 0 else None,
    )


def setup_tb_logger_trainer(
    tb_logger: TensorboardLogger,
    trainer: Engine,
    optimizer,
    output_handler_transform=lambda loss: {"batchloss": loss},
):

    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        param_name="lr",
        tag="train",
    )
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=4),
        tag="train",
        output_transform=output_handler_transform,
        metric_names="all",
    )
    return tb_logger


def setup_tb_logger_evaluator(
    tb_logger: TensorboardLogger,
    evaluator: Engine,
    metrics: tp.List[str],
    global_step_transform: tp.Callable,
    evaluator_name: tp.Optional[str] = None,
):
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=metrics,
        global_step_transform=global_step_transform,
    )
    # tb_logger.attach(
    #     evaluator,
    #     log_handler=log_figure(
    #         global_step_transform,
    #         "validation" + (f"/{evaluator_name}" if evaluator_name else ""),
    #     ),
    #     event_name=Events.ITERATION_COMPLETED(once=1),
    # )


def create_supervised_trainer_amp_grad_clip(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: tp.Union[tp.Callable[[tp.Any, tp.Any], torch.Tensor], torch.nn.Module],
    device: tp.Optional[tp.Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: tp.Callable = _prepare_batch,
    model_transform: tp.Callable[[tp.Any], tp.Any] = lambda output: output,
    output_transform: tp.Callable[
        [tp.Any, tp.Any, tp.Any, torch.Tensor], tp.Any
    ] = lambda x, y, y_pred, loss: loss.item(),
    scaler: tp.Optional["torch.cuda.amp.GradScaler"] = None,
    model_fn: tp.Callable[[torch.nn.Module, tp.Any], tp.Any] = lambda model, x: model(
        x
    ),
    grad_clipping_norm: float = 1.0,
) -> tp.Callable:
    def update(
        engine: Engine, batch: tp.Sequence[torch.Tensor]
    ) -> tp.Union[tp.Any, tp.Tuple[torch.Tensor]]:
        optimizer.zero_grad()
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        with autocast(enabled=True, dtype=torch.bfloat16):
            output = model_fn(model, x)
            y_pred = model_transform(output)
            loss = loss_fn(y_pred, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping_norm)
        scaler.step(optimizer)
        scaler.update()
        return output_transform(x, y, y_pred, loss)

    return Engine(update)
