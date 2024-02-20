# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/utils/benchmark.py

""" Functions for benchmarks. """

import numpy as np
import pprint
from tqdm import tqdm
from fvcore.common.timer import Timer
import torch

import floodsformer.utils.logg as logg
import floodsformer.utils.misc as misc
from floodsformer.datasets import loader
from floodsformer.utils.env import setup_environment

logger = logg.get_logger(__name__)


def benchmark_data_loading(cfg):
    """
    Benchmark the speed of data loading.
    Args:
        cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
    """
    # Set up environment.
    setup_environment()
    device = misc.get_device(cfg)
    # Set random seed from configs.
    misc.set_seed(cfg.RNG_SEED)

    # Setup logging format.
    logg.setup_logging(cfg.SAVE_RESULTS_PATH, cfg.NUM_GPUS * cfg.NUM_SHARDS)

    # Print config.
    logger.info("Benchmark data loading with config:")
    logger.info(pprint.pformat(cfg))

    timer = Timer()
    dataloader = loader.construct_loader(cfg, "train")
    logger.info("Initialize loader using {:.2f} seconds.".format(timer.seconds()))
    # Total batch size across different machines.
    batch_size = cfg.DATA.BATCH_SIZE * cfg.NUM_SHARDS
    log_period = cfg.BENCHMARK.LOG_PERIOD
    epoch_times = []
    # Test for a few epochs.
    for cur_epoch in range(cfg.BENCHMARK.NUM_EPOCHS):
        timer = Timer()
        timer_epoch = Timer()
        iter_times = []
        if cfg.BENCHMARK.SHUFFLE:
            loader.shuffle_dataset(dataloader, cur_epoch)
        for cur_iter, sample in enumerate(tqdm(dataloader, disable=True)):
            past_frames, future_frames = sample
            past_frames = past_frames.to(device)
            future_frames = future_frames.to(device)
            x = torch.cat([past_frames, future_frames], dim = 1)

            if cur_iter > 0 and cur_iter % log_period == 0:
                iter_times.append(timer.seconds())
                ram_usage, ram_total = misc.cpu_mem_usage()
                logger.info(
                    "Epoch {}: {} iters ({} videos) in {:.2f} seconds. "
                    "RAM Usage: {:.2f}/{:.2f} GB.".format(
                        cur_epoch,
                        log_period,
                        log_period * batch_size,
                        iter_times[-1],
                        ram_usage,
                        ram_total,
                    )
                )
                timer.reset()
        epoch_times.append(timer_epoch.seconds())
        ram_usage, ram_total = misc.cpu_mem_usage()
        logger.info(
            "Epoch {}: in total {} iters ({} videos) in {:.2f} seconds. "
            "RAM Usage: {:.2f}/{:.2f} GB.".format(
                cur_epoch,
                len(dataloader),
                len(dataloader) * batch_size,
                epoch_times[-1],
                ram_usage,
                ram_total,
            )
        )
        logger.info(
            "Epoch {}: on average every {} iters ({} videos) take {:.2f}/{:.2f} "
            "(avg/std) seconds.".format(
                cur_epoch,
                log_period,
                log_period * batch_size,
                np.mean(iter_times),
                np.std(iter_times),
            )
        )
    logger.info(
        "On average every epoch ({} videos) takes {:.2f}/{:.2f} "
        "(avg/std) seconds.".format(
            len(dataloader) * batch_size,
            np.mean(epoch_times),
            np.std(epoch_times),
        )
    )

    gpu_usage, gpu_total = misc.gpu_mem_usage()
    logger.info("GPU Usage: {:.2f}/{:.2f} GB.".format(gpu_usage, gpu_total))