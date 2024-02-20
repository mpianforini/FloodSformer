# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/tools/benchmark.py

""" A script to benchmark data loading. """

import torch
from floodsformer.utils.benchmark import benchmark_data_loading
from floodsformer.utils.misc import launch_job
from floodsformer.utils.parser import load_config, parse_args

def main():
    args = parse_args()
    cfg = load_config(args)

    print("\n[!] Results directory: ", cfg.SAVE_RESULTS_PATH)
    print("CUDA is available: ", torch.cuda.is_available())
    print("Num GPUs Available: ", torch.cuda.device_count())

    launch_job(cfg=cfg, init_method=args.init_method, func=benchmark_data_loading)

if __name__ == "__main__":
    main()
