""" Real-time forecasting with FloodSformer model (autoregressive procedure) """

import torch
import os
from datetime import datetime, timedelta

import floodsformer.utils.distributed as du
from floodsformer.models import build_AE_model, build_TS_model, RMSE, class_metrics
from floodsformer.models.helpers import RealTimeForecasting
from floodsformer.utils.misc import get_device, set_seed, gpu_mem_usage
from floodsformer.datasets.loader import construct_loader
import floodsformer.utils.logg as logg
from floodsformer.visualization.utils import map_to_image

logger = logg.get_logger(__name__)

def testFS(cfg):
    """
    Args:
        cfg (CfgNode): configs. Details can be found in: floodsformer/config/defaults.py
    """
    # Set up environment. Initialize variables needed for distributed training (NUM_GPUS > 1).
    du.init_distributed_training(cfg)
    device = get_device(cfg)
    rank = du.get_rank()
    word_size = du.get_world_size()
    # Set random seed from configs.
    set_seed(cfg.RNG_SEED)

    # Setup logging format.
    logg.setup_logging(cfg.SAVE_RESULTS_PATH, cfg.NUM_GPUS * cfg.NUM_SHARDS)
    logger.info("### Real-time forecasting ###")

    # Print config.
    logger.info("Configurations:\n{}".format(cfg))
    assert cfg.TRAIN.MODE == "train_VPTR", "Set cfg.TRAIN.MODE='train_VPTR' for the real-time forecasting procedure."

    ################### Init test dataset ########################
    test_loader = construct_loader(cfg, "AR_forecast")

    preprocessing_map = test_loader.dataset.get_preprocessing()
    print_image = map_to_image(cfg, preprocessing_map.extensions)

    ##################### Init Models and Optimizer ###########################
    VPTR_Enc, VPTR_Dec, _, _, _, _ = build_AE_model(cfg, device, pretrain=True, build_disc=False) # load AE pretrained model always

    # For the real-time forecasting application load always the VPTR model.
    assert os.path.isfile(cfg.MODEL.PRETRAINED_VPTR), "No pretrained VPTR model found at '{}'.".format(cfg.MODEL.PRETRAINED_VPTR)
    VPTR_Transformer, _, epoch, num_input_frames = build_TS_model(cfg, device, pretrain=True)

    if num_input_frames is not None:
        # For real-time forecasting the model must be equal to the trained one (no interpolation of parameters is allowed).
        assert num_input_frames == cfg.DATA.NUM_INPUT_FRAMES, "Load a pretrained model with number of input frames = {} but cfg.DATA.NUM_INPUT_FRAMES = {}!".format(num_input_frames, cfg.DATA.NUM_INPUT_FRAMES)

    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()
    VPTR_Transformer = VPTR_Transformer.eval()

    rmse_metric = RMSE(threshold=cfg.DATA.WET_DEPTH, device=device)
    metrics_classif = class_metrics(threshold=cfg.DATA.WET_DEPTH, device=device)

    ##################### Real-time forecasting ###########################
    save_dir = os.path.join(cfg.SAVE_RESULTS_PATH, "R-TForc_e{}_I{}_P{}_F{}".format(epoch - 1, cfg.DATA.NUM_INPUT_FRAMES, cfg.FORECAST.NUM_PAST_FRAMES, cfg.FORECAST.NUM_FUTURE_FRAMES))
    if not os.path.isdir(save_dir) and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        os.mkdir(save_dir)
    mean_rmse_app = None
    test_start = datetime.now()
    mod_fut_fram = cfg.DATA.NUM_INPUT_FRAMES + 1 - cfg.FORECAST.NUM_PAST_FRAMES

    logger.info("Run real-time forecasting")
    for iter, sample in enumerate(test_loader):
        mean_rmse = RealTimeForecasting(
            VPTR_Enc, 
            VPTR_Dec, 
            VPTR_Transformer, 
            sample, 
            save_dir, 
            device, 
            preprocessing_map, 
            print_image,
            rmse_metric,
            metrics_classif,
            iterXbatch=test_loader.batch_size * (rank + iter * word_size),
            mod_fut_fram=mod_fut_fram,
        )
        if cfg.NUM_GPUS > 1:
            mean_rmse = du.all_reduce([mean_rmse])[0]
        if mean_rmse_app is None:
            mean_rmse_app = mean_rmse.unsqueeze(0)
        else:
            mean_rmse_app = torch.cat((mean_rmse_app, mean_rmse.unsqueeze(0)), dim=0)

    mean_rmse_app = torch.mean(mean_rmse_app)
    test_time = datetime.now() - test_start

    if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        with open(os.path.join(save_dir, "rmse_stats.txt"), 'a') as f:
            f.write('\n\nMean RMSE wet: {:.4f}'.format(mean_rmse_app))

        logger.info(
            "End of real-time forecasting - Duration: {}. Mean RMSE wet: {:.4f} m; Max_gpu_mem: {:.2f}/{:.2f}G."
            .format(
                timedelta(seconds=int(test_time.total_seconds())), 
                mean_rmse_app,
                gpu_mem_usage()[0], 
                gpu_mem_usage()[1]
            )
        )