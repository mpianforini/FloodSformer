# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/datasets/loader.py

"""Data loader."""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from . import utils as utils
from .build import build_dataset

def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        split (str): the split of the data loader. Options include `train`, `val`, and `test`.
    """
    assert split in ["train", "val", "test", "AR_forecast"], "Split '{}' not supported for Parflood dataset.".format(split)

    if split in ["train"]:
        batch_size = int(cfg.DATA.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = False
        n = 0
    elif split in ["val"]:
        batch_size = int(cfg.DATA.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
        n = 1
    elif split in ["test"]:
        batch_size = int(cfg.DATA.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
        n = 2
    else: # AR_forecast
        batch_size = int(cfg.DATA.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
        n = 2

    # Construct the dataset
    dataset = build_dataset(cfg.DATA.DATASET_FILE, cfg, split, n, drop_last, batch_size)

    # Create a sampler for multi-process training
    sampler = utils.create_sampler(dataset, shuffle, cfg)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=None,
        worker_init_fn=utils.loader_worker_init_fn(dataset),
    )

    return loader

def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(sampler, (RandomSampler, DistributedSampler)), "Sampler type '{}' not supported".format(type(sampler))

    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
