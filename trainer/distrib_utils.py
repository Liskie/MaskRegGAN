import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def _dist_is_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_world_size() -> int:
    return dist.get_world_size() if _dist_is_available_and_initialized() else 1


def _get_rank() -> int:
    return dist.get_rank() if _dist_is_available_and_initialized() else 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _setup_ddp_if_needed(config) -> Tuple[bool, int]:
    """
    根据环境变量（torchrun 设置的 WORLD_SIZE/LOCAL_RANK）决定是否启用 DDP。
    config['ddp']=False 可强制关闭。
    """
    use_ddp = bool(config.get('ddp', True))
    if not use_ddp:
        return False, 0
    world_size_env = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size_env <= 1:
        return False, 0
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    return True, local_rank


def _reduce_tensor_sum(t: torch.Tensor) -> torch.Tensor:
    if not _dist_is_available_and_initialized():
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _enable_mc_dropout(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()  # activate dropout during inference
