import os
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers.checkpoint import Checkpoint, ModelCheckpoint
from ignite.handlers.ema_handler import EMAHandler
from ignite.handlers.param_scheduler import LRScheduler
from ignite.utils import setup_logger
from ml_mmr_tracking.data import (
    MeasuredNoiseTransform,
    MMRTrackingDataset,
    normalization,
    scale,
)
from ml_mmr_tracking.distributed import cleanup_distributed, setup_distributed
from ml_mmr_tracking.ignite import (
    attach_evaluator,
    create_supervised_trainer_amp_grad_clip,
    setup_metrics,
    setup_tb_logger_evaluator,
    setup_tb_logger_trainer,
)
from ml_mmr_tracking.models import model_factory
from ml_mmr_tracking.utils import seed_all
from omegaconf import DictConfig, OmegaConf
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn

SETTINGS["CAPTURE_MODE"] = "sys"

torch.backends.cudnn.benchmark = True


def prepare_batch(batch, device, non_blocking):
    signal, gt = batch
    signal = signal.to(device, non_blocking=non_blocking)
    gt = gt.to(device, non_blocking=non_blocking)
    return signal, gt


def create_dataset(config: DictConfig, mode: str) -> MMRTrackingDataset:
    data_folder = Path(config.data.folder)
    signal_postprocess_fn = {
        "train": v2.Compose(
            [
                normalization,
                MeasuredNoiseTransform(
                    "data/raw/training/noise.npy", noise_level=(0.0, 0.1)
                ),
                normalization,
                lambda s: torch.from_numpy(s).float().transpose(0, 1),
            ]
        ),
        "val": v2.Compose(
            [
                normalization,
                lambda s: torch.from_numpy(s).float().transpose(0, 1),
            ]
        ),
    }[mode]
    return MMRTrackingDataset(
        signal_fps=sorted(data_folder.glob(f"{mode}_chunk*.npy")),
        meta_csv_fp=data_folder / f"{mode}_meta.csv",
        signal_postprocess_fn=signal_postprocess_fn,
        gt_postprocess_fn=lambda g: torch.from_numpy(scale(g, factor=0.2)).float(),
        coordinate_column_names=("position_x", "position_y", "position_z"),
    )


def create_data_loaders(config, local_rank, world_size):
    """Create data loaders with distributed sampling if needed."""
    ds = {}
    for mode in config.data.modes:
        ds[mode] = create_dataset(config, mode)

    # Create samplers
    train_sampler = None
    val_sampler = None

    if world_size > 1:
        train_sampler = DistributedSampler(
            ds["train"], num_replicas=world_size, rank=local_rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            ds["val"], num_replicas=world_size, rank=local_rank, shuffle=False
        )

    train_dl = DataLoader(
        dataset=ds["train"],
        num_workers=config.data.modes.train.num_workers,
        batch_size=config.data.modes.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_dl = DataLoader(
        dataset=ds["val"],
        num_workers=config.data.modes.val.num_workers,
        batch_size=config.data.modes.val.batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_dl, val_dl, train_sampler


def train(config, seed, _run, local_rank, world_size, rank, device):
    """This is scared managed function that trains the model.
    _run comes from https://sacred.readthedocs.io/en/stable/configuration.html
    """
    config = OmegaConf.create(deepcopy(config))
    experiment_folder, models_folder = ".", "."
    if rank == 0:
        experiment_folder = Path(_run.observers[0].dir)
        models_folder = experiment_folder / "models"
        models_folder.mkdir(parents=True, exist_ok=True)

    seed_all(config.seed + rank)

    train_dl, val_dl, train_sampler = create_data_loaders(
        config, local_rank, world_size
    )
    model = model_factory.create_model("deep-conv-transformer").to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        # For EMA, we need the underlying model
        model_for_ema = model.module
    else:
        model_for_ema = model

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=1e-4
    )
    torch_lr_scheduler = MultiStepLR(
        optimizer,
        milestones=config.training.step_lr_scheduler.milestones,
        gamma=config.training.step_lr_scheduler.gamma,
    )
    scheduler = LRScheduler(torch_lr_scheduler)
    scaler = None
    if device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
        trainer = create_supervised_trainer_amp_grad_clip(
            model,
            optimizer,
            criterion,
            device=device,
            non_blocking=True,
            prepare_batch=prepare_batch,
            scaler=scaler,
            grad_clipping_norm=config.training.grad_clipping_norm,
        )
    else:
        trainer = create_supervised_trainer(
            model,
            optimizer,
            criterion,
            device=device,
            prepare_batch=prepare_batch,
        )

    trainer.logger = setup_logger("trainer", distributed_rank=rank)

    if config.training.ema.enabled:
        ema_handler = EMAHandler(
            model_for_ema,
            momentum=config.training.ema.momentum,
            warmup_iters=config.training.ema.warmup_iters,
            momentum_warmup=config.training.ema.momentum_warmup,
        )
        ema_model = ema_handler.ema_model
        ema_handler.attach(
            trainer,
            name="ema_momentum",
            event=Events.ITERATION_COMPLETED(every=config.training.ema.update_every),
        )

    if config.training.resume_from is not None:
        checkpoint_path = Path(config.training.resume_from)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Prepare objects to load
        to_load = {
            "model": model_for_ema,
            "trainer": trainer,
            "optimizer": optimizer,
        }

        if config.training.ema.enabled and "ema_model" in checkpoint:
            to_load["ema_model"] = ema_model

        if "scheduler" in checkpoint:
            to_load["scheduler"] = torch_lr_scheduler

        if scaler is not None and "scaler" in checkpoint:
            to_load["scaler"] = scaler

        Checkpoint.load_objects(
            to_load=to_load,
            checkpoint=checkpoint,
        )
        print(f"✔️ Resumed from checkpoint: {checkpoint_path}")
        print(f"✔️ Resumed from epoch: {trainer.state.epoch}")
        print(f"✔️ Resumed from iteration: {trainer.state.iteration}")

    if world_size > 1:
        trainer.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: train_sampler.set_epoch(engine.state.epoch),
        )

    if config.training.step_lr_scheduler.enabled:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Checkpoints
    to_save = {
        "model": model_for_ema,
        "trainer": trainer,
        "optimizer": optimizer,
        "scheduler": torch_lr_scheduler,
    }
    if config.training.ema.enabled:
        to_save["ema_model"] = ema_model
    if scaler is not None:
        to_save["scaler"] = scaler
    checkpoint_handler = ModelCheckpoint(
        dirname=models_folder,
        filename_prefix="ckpt",
        n_saved=5,
        global_step_transform=global_step_from_engine(trainer),
        create_dir=True,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=10),
        checkpoint_handler,
        to_save,
    )

    evaluators = []
    evaluator_metrics = {}

    evaluator = create_supervised_evaluator(
        ema_model if config.training.ema.enabled else model_for_ema,
        device=device,
        prepare_batch=prepare_batch,
    )
    evaluator.logger = setup_logger("val", distributed_rank=rank)
    metrics = setup_metrics(evaluator, suffix="-val")
    evaluator_metrics["val"] = metrics
    rmse_key = next((key for key in metrics if "rmse" in key.lower()), None)
    if rmse_key is None:
        raise ValueError(
            f"No RMSE metric found for val. Available metrics: {list(metrics)}"
        )
    attach_evaluator(
        trainer,
        evaluator,
        val_dl,
        config.training.validate_every_iteration,
    )
    evaluators.append(("val", evaluator))
    checkpoint_handler = ModelCheckpoint(
        models_folder,
        score_name=rmse_key,
        score_function=lambda engine, key=rmse_key: engine.state.metrics[key],
        n_saved=5,
        global_step_transform=lambda engine, _: global_step_from_engine(trainer)(
            engine, Events.ITERATION_COMPLETED
        ),
        greater_or_equal=True,
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, to_save)

    if rank == 0:
        # Tensorboard Logger
        tb_logger = TensorboardLogger(log_dir=experiment_folder)
        setup_tb_logger_trainer(tb_logger, trainer, optimizer)
        for name, evaluator in evaluators:
            setup_tb_logger_evaluator(
                tb_logger,
                evaluator,
                evaluator_metrics[name],
                global_step_transform=lambda engine, _: global_step_from_engine(
                    trainer
                )(engine, Events.ITERATION_COMPLETED),
                evaluator_name=name,
            )

        if device.startswith("cuda"):
            GpuInfo().attach(trainer, name="gpu")

        ProgressBar().attach(trainer, output_transform=lambda x: {"loss": x})

    trainer.run(train_dl, config.training.epochs)

    if rank == 0:
        tb_logger.close()


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Check if we should run distributed
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # Distributed training
        print(f"Starting distributed training with {world_size} processes")
        local_rank, world_size, rank, device = setup_distributed()
        try:
            # Only rank 0 sets up Sacred experiment
            if rank == 0:
                ex = Experiment(base_dir="..")
                ex.observers.append(FileStorageObserver(cfg.base_models_folder))
                ex.add_config({"config": OmegaConf.to_object(cfg), "seed": cfg.seed})
                ex.main(
                    lambda config, seed, _run: train(
                        config, seed, _run, local_rank, world_size, rank, device
                    )
                )
                ex.run()
            else:
                train(
                    OmegaConf.to_object(cfg),
                    cfg.seed,
                    None,
                    local_rank,
                    world_size,
                    rank,
                    device,
                )
        finally:
            cleanup_distributed()
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Single GPU/CPU training
        print("Starting single process training on device:", device)
        ex = Experiment(base_dir="..")
        ex.observers.append(FileStorageObserver(cfg.base_models_folder))
        ex.add_config({"config": OmegaConf.to_object(cfg), "seed": cfg.seed})
        ex.main(lambda config, seed, _run: train(config, seed, _run, 0, 1, 0, device))
        ex.run()


if __name__ == "__main__":
    main()
