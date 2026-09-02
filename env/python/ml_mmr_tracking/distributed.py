import datetime
import os

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training properly"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        # Initialize process group with longer timeout
        torch.distributed.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=1800)  # 30 minutes
        )

        # Add synchronization barrier
        torch.distributed.barrier()

        return rank, world_size, local_rank, f"cuda:{local_rank}"
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
