# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/utils/misc.py

import numpy as np
import psutil
import torch
import random
from torch import Tensor
from typing import Optional

import floodsformer.utils.logg as logg
import floodsformer.utils.multiprocess as mpu

logger = logg.get_logger(__name__)

def get_device(cfg, gpu_id=None):
    """
    Find the current device.
    Args:
        cfg (configs): configs. Details can be seen in floodsformer/config/defaults.py
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
    else:
        cur_device = torch.device('cpu')

    return cur_device

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    Returns:
        mem_usage (float): GPU used memory (GB).
        mem_total (float): GPU total memory (GB).
    """
    if torch.cuda.is_available():
        mem_usage = torch.cuda.max_memory_allocated() / 1024 ** 3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        mem_usage = 0.0
        mem_total = 0.0
    return mem_usage, mem_total

def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total

def log_model_info(models):
    """
    Print the number of parameters of the model(s).
    Args:
        models (dict): models to log the info.
    """
    for name in models:
        param = sum(p.numel() for p in models[name].parameters() if p.requires_grad)
        logger.info("{} num_parameters: {:,}".format(name, param))

def is_eval_epoch(cur_epoch, max_epoch, eval_period):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cur_epoch (int): current epoch.
        max_epoch (int): last epoch.
        eval_period (int): evaluate model on validation dataset every eval_period epochs.
    """
    if cur_epoch + 1 == max_epoch:
        return True
    return cur_epoch % eval_period == 0

def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg.
    Args:
        cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        init_method (str): initialization method to launch the job with multiple devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True, daemonic
                       processes will be created.
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
