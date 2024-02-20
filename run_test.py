# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/tools/run_net.py

""" Wrapper to test the autoregressive prediction of the FloodSformer (FS) model (real-time forecasting). """

from floodsformer.utils.misc import launch_job
from floodsformer.utils.parser import load_config, parse_args

import torch
from tools.test_FS import testFS

def main():
    """ Main function to spawn the real-time forecasting process. """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    print("\n[!] Results directory: ", cfg.SAVE_RESULTS_PATH)
    print("CUDA is available: ", torch.cuda.is_available())
    print("Num GPUs Available: ", torch.cuda.device_count())

    # Perform test of the FloodSformer model.
    launch_job(cfg=cfg, init_method=args.init_method, func=testFS)

if __name__ == "__main__":
    main()
