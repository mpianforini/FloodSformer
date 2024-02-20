# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/tools/run_net.py

""" Wrapper to train and test the FloodSformer (FS) model. """

from floodsformer.utils.misc import launch_job
from floodsformer.utils.parser import load_config, parse_args

import torch
from tools.train_AE import trainAE
from tools.train_VPTR import trainVPTR

def get_func():
    train_AE = trainAE
    train_VPTR = trainVPTR
    return train_AE, train_VPTR

def main():
    """ Main function to spawn the train and test process. """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    print("\n[!] Results directory: ", cfg.SAVE_RESULTS_PATH)
    print("CUDA is available: ", torch.cuda.is_available())
    print("Num GPUs Available: ", torch.cuda.device_count())

    train_AE, train_VPTR = get_func()

    # Perform AE training.
    if cfg.TRAIN.MODE == "train_AE":
        launch_job(cfg=cfg, init_method=args.init_method, func=train_AE)

    # Perform FloodSformer training (use pretrained AE parameters).
    if cfg.TRAIN.MODE == "train_VPTR":
        launch_job(cfg=cfg, init_method=args.init_method, func=train_VPTR)

if __name__ == "__main__":
    main()
