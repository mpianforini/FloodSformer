"""
Train the VPTR module of the FloodSformer (FS) model.
Modified based on VPTR-FAR model: https://github.com/XiYe20/VPTR/blob/main/train_FAR.py
"""

import torch
import torch.nn as nn
import os
from datetime import datetime, timedelta

import floodsformer.utils.distributed as du
import floodsformer.visualization.tensorboard_vis as tb
from floodsformer.models import build_AE_model, build_TS_model, GDL, MSELoss, L1Loss, RMSE, class_metrics
from floodsformer.models.helpers import EarlyStopper, init_loss_dict, test_show_sample, val_show_sample, RealTimeForecasting
from floodsformer.utils import misc
from floodsformer.utils.checkpoint import resume_train_transformer, save_ckpt
from floodsformer.utils.meters import AverageMeters
from floodsformer.utils.losses import cal_lossT
from floodsformer.datasets import loader
import floodsformer.utils.logg as logg
from floodsformer.utils.lr_policy import lr_scheduler
from floodsformer.visualization.utils import map_to_image

logger = logg.get_logger(__name__)

def train_single_iter(
        VPTR_Enc, 
        VPTR_Dec, 
        VPTR_Transformer, 
        optimizer_T, 
        sample, 
        device, 
        mse_loss,
        gdl_loss,
        l1_loss,
        cfg,
        scaler=None,
    ):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)

    with torch.no_grad():
        x = torch.cat([past_frames, future_frames[:, 0:-1, ...]], dim = 1)
        gt_feats = VPTR_Enc(x)

    VPTR_Transformer.zero_grad(set_to_none=True)
    VPTR_Dec.zero_grad(set_to_none=True)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        pred_future_feats = VPTR_Transformer(gt_feats)
        pred_frames = VPTR_Dec(pred_future_feats)

    # update Transformer (generator)
    loss_T, T_GDL_loss, T_MSE_loss, T_L1_loss = cal_lossT(pred_frames, torch.cat([past_frames[:, 1:, ...], future_frames], dim = 1), mse_loss, gdl_loss, l1_loss, cfg)

    iter_loss_dict = {'T_L1': T_L1_loss.item(), 'T_total': loss_T.item(), 'T_MSE': T_MSE_loss.item(), 'T_GDL': T_GDL_loss.item()}

    scaler.scale(loss_T).backward()
    nn.utils.clip_grad_norm_(VPTR_Transformer.parameters(), max_norm=1.0, norm_type=2)
    scaler.step(optimizer_T)

    scaler.update()

    return iter_loss_dict

def val_single_iter(
        VPTR_Enc, 
        VPTR_Dec, 
        VPTR_Transformer, 
        sample, 
        device,
        rmse_metric,
        mse_loss,
        gdl_loss,
        l1_loss,
        preprocessing_map,
        print_image,
        iterXbatch,
        save_dir,
        cfg,
    ):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)

    with torch.no_grad():
        gt_feats = VPTR_Enc(past_frames)
        pred_future_feats = VPTR_Transformer(gt_feats)
        pred_frames = VPTR_Dec(pred_future_feats)

        loss_T, T_GDL_loss, T_MSE_loss, T_L1_loss = cal_lossT(pred_frames, torch.cat([past_frames[:, 1:, ...], future_frames], dim = 1), mse_loss, gdl_loss, l1_loss, cfg)
        if cfg.NUM_GPUS > 1:
            iter_loss_dict = {'T_L1': T_L1_loss, 'T_total': loss_T, 'T_MSE': T_MSE_loss, 'T_GDL': T_GDL_loss}
        else:
            iter_loss_dict = {'T_L1': T_L1_loss.item(), 'T_total': loss_T.item(), 'T_MSE': T_MSE_loss.item(), 'T_GDL': T_GDL_loss.item()}

        rmse_all, rmse_wet = val_show_sample(
            past_frames, 
            future_frames, 
            pred_frames, 
            save_dir, 
            preprocessing_map, 
            print_image,
            rmse_metric, 
            iterXbatch=iterXbatch, 
        )

    return iter_loss_dict, rmse_all, rmse_wet

def trainVPTR(cfg):
    """
    Train VPTR module on train dataset and evaluate it on test dataset.
    The autoencoder (AE) model use pretrained parameters.
    Based on VPTR-FAR model: https://github.com/XiYe20/VPTR/blob/main/train_FAR.py
    Args:
        cfg (CfgNode): configs. Details can be found in: floodsformer/config/defaults.py
    """

    # Set up environment. Initialize variables needed for distributed training (NUM_GPUS > 1).
    du.init_distributed_training(cfg)
    device = misc.get_device(cfg)
    rank = du.get_rank()
    word_size = du.get_world_size()
    # Set random seed from configs.
    misc.set_seed(cfg.RNG_SEED)

    # Setup logging format.
    logg.setup_logging(cfg.SAVE_RESULTS_PATH, cfg.NUM_GPUS * cfg.NUM_SHARDS)
    logger.info("### Train FloodSformer model ###")

    # Print config.
    logger.info("Configurations:\n{}".format(cfg))

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        summary_writer = tb.TensorboardWriter(cfg)
    else:
        summary_writer = None

    # Initialize the early stopping function.
    early_stopping = EarlyStopper(patience=cfg.TRAIN.PATIENCE)

    # Initialize the learning rate scheduler.
    LRscheduler = lr_scheduler(cfg)

    ################### Init train and val dataset ########################
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    preprocessing_map = train_loader.dataset.dataset.get_preprocessing()
    print_image = map_to_image(cfg, preprocessing_map.extensions)

    ##################### Init Models and Optimizer ###########################
    VPTR_Enc, VPTR_Dec, _, _, _, _ = build_AE_model(cfg, device, pretrain=True, build_disc=False) # load pretrained model always
    VPTR_Transformer, optimizer_T, start_epoch, _ = build_TS_model(cfg, device, pretrain=cfg.MODEL.PRETRAIN)

    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()

    # Print model statistics.
    if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        misc.log_model_info({'Transformer': VPTR_Transformer})

    ##################### Init loss function ###########################
    loss_name_list = ['T_L1', 'T_MSE', 'T_GDL', 'T_total']

    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss().to(device, non_blocking=True)
    gdl_loss = GDL(alpha = 1).to(device, non_blocking=True)
    l1_loss = L1Loss().to(device, non_blocking=True)
    rmse_metric = RMSE(threshold=cfg.DATA.WET_DEPTH, device=device)

    ##################### Training loop ################################
    scaler = torch.cuda.amp.GradScaler()
    iter = 0

    if cfg.TRAIN.CHECKPOINT_EPOCH_RESET:
        start_epoch = 0

    if start_epoch < cfg.SOLVER.MAX_EPOCH and cfg.TRAIN.ENABLE:  
        logger.info("--- Train model from epoch {} to epoch {} ---".format(start_epoch, cfg.SOLVER.MAX_EPOCH - 1))
        train_start = datetime.now()
    else:
        epoch = start_epoch - 1
        start_epoch = cfg.SOLVER.MAX_EPOCH  # to skip the training if cfg.TRAIN.ENABLE==False

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        logger.info("Start epoch {}".format(epoch))
        epoch_st = datetime.now()

        # Update the learning rate.
        _ = LRscheduler(epoch, [optimizer_T], summary_writer)

        #Train
        EpochAveMeter = AverageMeters(loss_name_list)
        VPTR_Transformer.train()
        VPTR_Dec.eval()

        for _, sample in enumerate(train_loader):
            iter_loss_dict = train_single_iter(
                VPTR_Enc, 
                VPTR_Dec, 
                VPTR_Transformer, 
                optimizer_T, 
                sample, 
                device, 
                mse_loss,
                gdl_loss,
                l1_loss,
                cfg,
                scaler=scaler,
            )
            EpochAveMeter.iter_update(iter_loss_dict)

        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = True)
        if summary_writer is not None:
            summary_writer.write_summary(loss_dict, train_flag = True)

        logger.info("TRAIN - T_total: {:.3e}.".format(loss_dict["T_total"].train))

        if misc.is_eval_epoch(epoch, cfg.SOLVER.MAX_EPOCH, cfg.TRAIN.EVAL_PERIOD):
            # validation
            EpochAveMeter = AverageMeters(loss_name_list)
            rmse_all_sum = 0
            rmse_wet_sum = 0

            VPTR_Transformer.eval()
            VPTR_Dec.eval()

            save_dir = os.path.join(cfg.SAVE_RESULTS_PATH, "val_epoch{}".format(epoch))
            if not os.path.isdir(save_dir) and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
                os.mkdir(save_dir)

            for iter, sample in enumerate(val_loader):
                iter_loss_dict, rmse_all, rmse_wet = val_single_iter(
                    VPTR_Enc, 
                    VPTR_Dec, 
                    VPTR_Transformer, 
                    sample, 
                    device, 
                    rmse_metric, 
                    mse_loss,
                    gdl_loss,
                    l1_loss,
                    preprocessing_map,
                    print_image,
                    val_loader.batch_size * (rank + iter * word_size),
                    save_dir=save_dir,
                    cfg=cfg,
                )
                if cfg.NUM_GPUS > 1:
                    for key in iter_loss_dict.keys():
                        iter_loss_dict[key] = du.all_reduce([iter_loss_dict[key]])[0].item()
                    rmse_all, rmse_wet = du.all_reduce([rmse_all, rmse_wet])
                EpochAveMeter.iter_update(iter_loss_dict)
                rmse_all_sum += rmse_all
                rmse_wet_sum += rmse_wet

            mean_rmse_all = rmse_all_sum / (iter + 1)
            mean_rmse_wet = rmse_wet_sum / (iter + 1)
            loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = False)
            logger.info("VAL - T_total: {:.3e}.".format(loss_dict["T_total"].val))
            logger.info("VAL - RMSE all: {:.4f} m; RMSE wet: {:.4f} m".format(mean_rmse_all, mean_rmse_wet))

            if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
                if summary_writer is not None:
                    summary_writer.add_scalars({"val_RMSE_all": mean_rmse_all, "val_RMSE_wet": mean_rmse_wet}, epoch)
                    summary_writer.write_summary(loss_dict, train_flag = False)

                with open(os.path.join(save_dir, "rmse_stats.txt"), 'a') as f:
                    f.write('\nVAL: mean RMSE all maps: {:.4f} m'.format(mean_rmse_all))
                    f.write('\tmean RMSE wet cells: {:.4f} m'.format(mean_rmse_wet))

            # Early stopping.
            if early_stopping(mean_rmse_wet, epoch):
                logger.info("Early stopping! Epoch of the last RMSE improvement: {}".format(early_stopping.best_epoch))
                epoch = early_stopping.best_epoch
                # Load the best weights
                _, _ = resume_train_transformer(
                    module_dict={'VPTR_Transformer': VPTR_Transformer},
                    optimizer_dict={},
                    resume_ckpt=os.path.join(cfg.CHECKPOINT_PATH, "epoch_{}.tar".format(epoch)), 
                    map_location='cpu' if cfg.NUM_GPUS == 0 else None,
                    gpu_correction=cfg.NUM_GPUS,
                )
                break
            elif early_stopping.counter == 0: # RMSE improved
                # Remove old checkpoint
                if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
                    for filename in os.listdir(cfg.CHECKPOINT_PATH):
                        os.remove(os.path.join(cfg.CHECKPOINT_PATH, filename))
                    # Save new checkpoint
                    save_ckpt({'VPTR_Transformer': VPTR_Transformer}, {'optimizer_T': optimizer_T}, epoch, cfg.CHECKPOINT_PATH, cfg.DATA.NUM_INPUT_FRAMES)
            else:
                logger.info("Early stopping: no improvement (count: {}).".format(early_stopping.counter))

        epoch_time = datetime.now() - epoch_st
        time_from_start = datetime.now() - train_start
        logger.info(
            "End epoch {}. Duration: {}. Eta training: {}"
            .format(
                epoch, 
                timedelta(seconds=int(epoch_time.total_seconds())), 
                timedelta(seconds=int(time_from_start.total_seconds() * (cfg.SOLVER.MAX_EPOCH - epoch - 1) / (epoch - start_epoch + 1)))
            )
        )
    # ------------

    if start_epoch < cfg.SOLVER.MAX_EPOCH: 
        train_time = datetime.now() - train_start
        eta_sec = int(train_time.total_seconds())
        logger.info("End of training. Duration: {}. Max_gpu_mem: {:.2f}/{:.2f}G.".format(timedelta(seconds=eta_sec), misc.gpu_mem_usage()[0], misc.gpu_mem_usage()[1]))

        if early_stopping.best_epoch != epoch:
            logger.info("Epoch of the last RMSE improvement: {}".format(early_stopping.best_epoch))
            epoch = early_stopping.best_epoch
            # Load the best weights
            _ = resume_train_transformer(
                module_dict={'VPTR_Transformer': VPTR_Transformer},
                optimizer_dict={},
                resume_ckpt=os.path.join(cfg.CHECKPOINT_PATH, "epoch_{}.tar".format(epoch)), 
                map_location='cpu' if cfg.NUM_GPUS == 0 else None,
                gpu_correction=cfg.NUM_GPUS,
            )

    ##################### Final VPTR test ###########################
    VPTR_Transformer = VPTR_Transformer.eval()
    metrics_classif = class_metrics(threshold=cfg.DATA.WET_DEPTH, device=device)

    if cfg.TEST.ENABLE:
        logger.info(" ")
        logger.info("--- Run VPTR test ---")
        # Init test dataset
        test_loader = loader.construct_loader(cfg, "test")
        preprocessing_map = test_loader.dataset.get_preprocessing()

        save_dir = os.path.join(cfg.SAVE_RESULTS_PATH, "testVPTR_epoch{}_I{}".format(epoch, cfg.DATA.NUM_INPUT_FRAMES))
        if not os.path.isdir(save_dir) and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            os.mkdir(save_dir)
        mean_rmse_app = None
        test_start = datetime.now()
        for iter, sample in enumerate(test_loader):
            mean_rmse, mean_class = test_show_sample(
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
            )
            if cfg.NUM_GPUS > 1:
                mean_rmse, mean_class = du.all_reduce([mean_rmse, mean_class])
            if mean_rmse_app is None:
                mean_rmse_app = mean_rmse.unsqueeze(0)
                mean_class_app = mean_class.unsqueeze(0)
            else:
                mean_rmse_app = torch.cat((mean_rmse_app, mean_rmse.unsqueeze(0)), dim=0)
                mean_class_app = torch.cat((mean_class_app, mean_class.unsqueeze(0)), dim=0)
        mean_rmse_app = torch.mean(mean_rmse_app, dim=0)
        mean_class_app = torch.mean(mean_class_app, dim=0)
        test_time = datetime.now() - test_start

        if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            with open(os.path.join(save_dir, "rmse_stats.txt"), 'a') as f:
                f.write('\nTEST - mean RMSE wet: ')
                for i in range(mean_rmse_app.shape[0]):
                    f.write('t={}: {:.4f} m\t'.format(i + 1, mean_rmse_app[i]))
                f.write('\n\nMean RMSE wet: {:.4f}; Mean precision: {:.3f}; Mean recall: {:.3f}; Mean F1: {:.3f}'.format(torch.mean(mean_rmse_app), mean_class_app[0], mean_class_app[1], mean_class_app[2]))

            logger.info(
                "No autoregressive test - Duration: {}. Mean RMSE wet: {:.4f} m; Mean F1: {:.3f}; Max_gpu_mem: {:.2f}/{:.2f}G."
                .format(
                    timedelta(seconds=int(test_time.total_seconds())), 
                    torch.mean(mean_rmse_app),
                    mean_class_app[2],
                    misc.gpu_mem_usage()[0], 
                    misc.gpu_mem_usage()[1]
                )
            )

    ##################### Real-time forecasting ###########################
    if cfg.FORECAST.ENABLE:
        logger.info(" ")
        logger.info("--- Run real-time forecasting (autoregressive) ---")
        # Init test dataset
        test_loader = loader.construct_loader(cfg, "AR_forecast")
        preprocessing_map = test_loader.dataset.get_preprocessing()

        save_dir = os.path.join(cfg.SAVE_RESULTS_PATH, "R-TForc_e{}_I{}_P{}_F{}".format(epoch, cfg.DATA.NUM_INPUT_FRAMES, cfg.FORECAST.NUM_PAST_FRAMES, cfg.FORECAST.NUM_FUTURE_FRAMES))
        if not os.path.isdir(save_dir) and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            os.mkdir(save_dir)
        mean_rmse_app = None
        test_start = datetime.now()
        mod_fut_fram = cfg.DATA.NUM_INPUT_FRAMES + 1 - cfg.FORECAST.NUM_PAST_FRAMES
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
                    misc.gpu_mem_usage()[0], 
                    misc.gpu_mem_usage()[1]
                )
            )

    if summary_writer is not None:
        summary_writer.close()