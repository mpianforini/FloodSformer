"""
Train the AutoEncoder (AE) module of the FloodSformer (FS) model.
Modified based on VPTR-AE model: https://github.com/XiYe20/VPTR/blob/main/train_AutoEncoder.py
"""

import torch
from datetime import datetime, timedelta
import os

import floodsformer.utils.distributed as du
import floodsformer.utils.logg as logg
from floodsformer.utils import misc
import floodsformer.visualization.tensorboard_vis as tb
from floodsformer.datasets import loader
from floodsformer.utils.meters import AverageMeters
from floodsformer.utils.checkpoint import is_checkpoint_epoch, save_ckpt
from floodsformer.models import build_AE_model, GDL, MSELoss, GANLoss, RMSE, class_metrics, L1Loss
from floodsformer.utils.losses import cal_lossD, cal_lossG
from floodsformer.models.helpers import init_loss_dict, test_show_samples_AE, val_show_samples_AE
from floodsformer.utils.lr_policy import lr_scheduler
from floodsformer.visualization.utils import map_to_image

logger = logg.get_logger(__name__)

def train_single_iter(
        VPTR_Enc, 
        VPTR_Dec, 
        VPTR_Disc, 
        optimizer_G, 
        optimizer_D, 
        gt_frames, 
        device, 
        gan_loss,
        mse_loss,
        gdl_loss,
        l1_loss,
        cfg,
        scaler=None,
    ):
    gt_frames = gt_frames.to(device, non_blocking=True)

    VPTR_Enc.zero_grad(set_to_none=True)
    VPTR_Dec.zero_grad(set_to_none=True)

    if cfg.FLOODSFORMER.LAM_GAN != 0.0:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            #update discriminator
            VPTR_Disc.train()
            for p in VPTR_Disc.parameters():
                p.requires_grad_(True)
            VPTR_Disc.zero_grad(set_to_none=True)

            rec_frames = VPTR_Dec(VPTR_Enc(gt_frames))

            loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, rec_frames, gt_frames, cfg.FLOODSFORMER.LAM_GAN, gan_loss)
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            #update autoencoder (generator)
            for p in VPTR_Disc.parameters():
                p.requires_grad_(False)
            loss_G, loss_G_gan, AE_MSE_loss, AE_GDL_loss, AE_L1_loss = cal_lossG(VPTR_Disc, rec_frames, gt_frames, gan_loss, mse_loss, gdl_loss, l1_loss, cfg)
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

        iter_loss_dict = {'AE_L1': AE_L1_loss.item(), 'AEgan': loss_G_gan.item(), 'AE_MSE': AE_MSE_loss.item(), 'AE_GDL': AE_GDL_loss.item(), 'AE_total': loss_G.item(), 'Dtotal': loss_D.item(), 'Dfake':loss_D_fake.item(), 'Dreal':loss_D_real.item()}
    else:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            rec_frames = VPTR_Dec(VPTR_Enc(gt_frames))
            loss_G, _, AE_MSE_loss, AE_GDL_loss, AE_L1_loss = cal_lossG(VPTR_Disc, rec_frames, gt_frames, gan_loss, mse_loss, gdl_loss, l1_loss, cfg)
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

        iter_loss_dict = {'AE_L1': AE_L1_loss.item(), 'AE_MSE': AE_MSE_loss.item(), 'AE_GDL': AE_GDL_loss.item(), 'AE_total': loss_G.item()}

    return iter_loss_dict

def val_single_iter(
        VPTR_Enc, 
        VPTR_Dec, 
        VPTR_Disc, 
        gt_frames, 
        device, 
        rmse_metric,
        metrics_classif,
        gan_loss,
        mse_loss,
        gdl_loss,
        l1_loss,
        cfg,
        preprocessing_map,
        print_image,
        cur_iter,
        save_dir,
    ):
    gt_frames = gt_frames.to(device)

    with torch.no_grad():
        rec_frames = VPTR_Dec(VPTR_Enc(gt_frames))
        if cfg.FLOODSFORMER.LAM_GAN != 0.0:
            loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, rec_frames, gt_frames, cfg.FLOODSFORMER.LAM_GAN, gan_loss)
            loss_G, loss_G_gan, AE_MSE_loss, AE_GDL_loss, AE_L1_loss = cal_lossG(VPTR_Disc, rec_frames, gt_frames, gan_loss, mse_loss, gdl_loss, l1_loss, cfg)
            if cfg.NUM_GPUS > 1:
                iter_loss_dict = {'AE_L1': AE_L1_loss, 'AEgan': loss_G_gan, 'AE_MSE': AE_MSE_loss, 'AE_GDL': AE_GDL_loss, 'AE_total': loss_G, 'Dtotal': loss_D, 'Dfake':loss_D_fake, 'Dreal':loss_D_real}
            else:
                iter_loss_dict = {'AE_L1': AE_L1_loss.item(), 'AEgan': loss_G_gan.item(), 'AE_MSE': AE_MSE_loss.item(), 'AE_GDL': AE_GDL_loss.item(), 'AE_total': loss_G.item(), 'Dtotal': loss_D.item(), 'Dfake':loss_D_fake.item(), 'Dreal':loss_D_real.item()}
        else:
            loss_G, _, AE_MSE_loss, AE_GDL_loss, AE_L1_loss = cal_lossG(VPTR_Disc, rec_frames, gt_frames, gan_loss, mse_loss, gdl_loss, l1_loss, cfg)
            if cfg.NUM_GPUS > 1:
                iter_loss_dict = {'AE_L1': AE_L1_loss, 'AE_MSE': AE_MSE_loss, 'AE_GDL': AE_GDL_loss, 'AE_total': loss_G}
            else:
                iter_loss_dict = {'AE_L1': AE_L1_loss.item(), 'AE_MSE': AE_MSE_loss.item(), 'AE_GDL': AE_GDL_loss.item(), 'AE_total': loss_G.item()}

        rmse_all, rmse_wet, f1_metric = val_show_samples_AE(
            gt_frames, 
            rec_frames, 
            save_dir, 
            preprocessing_map, 
            print_image,
            rmse_metric,
            metrics_classif,
            cur_iter
        )

    return iter_loss_dict, rmse_all, rmse_wet, f1_metric

def trainAE(cfg):
    """
    Train AutoEncoder (AE) model on train dataset and evaluate it on test dataset.
    Based on VPTR-AE model: https://github.com/XiYe20/VPTR/blob/main/train_AutoEncoder.py
    Args:
        cfg (CfgNode): configs. Details can be found in: floodsformer/config/defaults.py
    """

    # Set up environment. Initialize variables needed for distributed training (NUM_GPUS > 1)
    du.init_distributed_training(cfg)
    device = misc.get_device(cfg)
    rank = du.get_rank()
    word_size = du.get_world_size()
    # Set random seed from configs.
    misc.set_seed(cfg.RNG_SEED)

    # Setup logging format.
    logg.setup_logging(cfg.SAVE_RESULTS_PATH, cfg.NUM_GPUS * cfg.NUM_SHARDS)
    logger.info("### Train AutoEncoder model ###")

    # Print config.
    logger.info("Configurations:\n{}".format(cfg))

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        summary_writer = tb.TensorboardWriter(cfg)
    else:
        summary_writer = None

    # Initialize the learning rate scheduler.
    LRscheduler = lr_scheduler(cfg)

    ##################### Init Models and Optimizer ###########################
    VPTR_Enc, VPTR_Dec, VPTR_Disc, optimizer_D, optimizer_G, start_epoch = build_AE_model(cfg, device, pretrain=cfg.MODEL.PRETRAIN, build_disc=True)

    # Print model statistics.
    if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        misc.log_model_info({'Encoder': VPTR_Enc, 'Decoder': VPTR_Dec, 'Discriminator': VPTR_Disc})

    ##################### Init loss function ###########################
    if cfg.FLOODSFORMER.LAM_GAN != 0.0:
        loss_name_list = ['AE_L1', 'AE_MSE', 'AE_GDL', 'AE_total', 'Dtotal', 'Dfake', 'Dreal', 'AEgan']
        gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device, non_blocking=True)
    else:
        loss_name_list = ['AE_L1', 'AE_MSE', 'AE_GDL', 'AE_total']
        gan_loss = None
 
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss().to(device, non_blocking=True)
    gdl_loss = GDL(alpha = 1).to(device, non_blocking=True)
    l1_loss = L1Loss().to(device, non_blocking=True)
    rmse_metric = RMSE(threshold=cfg.DATA.WET_DEPTH, device=device)
    metrics_classif = class_metrics(threshold=cfg.DATA.WET_DEPTH, device=device)

    ################### Init train and val dataset ########################
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    preprocessing_map = train_loader.dataset.dataset.get_preprocessing()
    print_image = map_to_image(cfg, preprocessing_map.extensions)

    ##################### Training loop ###########################
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
        _ = LRscheduler(epoch, [optimizer_D, optimizer_G], summary_writer)

        # Train
        EpochAveMeter = AverageMeters(loss_name_list)
        VPTR_Enc.train()
        VPTR_Dec.train()

        for _, sample in enumerate(train_loader, 0):
            iter_loss_dict = train_single_iter(
                VPTR_Enc, 
                VPTR_Dec, 
                VPTR_Disc,
                optimizer_G, 
                optimizer_D, 
                sample[1], 
                device, 
                gan_loss,
                mse_loss,
                gdl_loss,
                l1_loss,
                cfg=cfg,
                scaler=scaler,
            )
            EpochAveMeter.iter_update(iter_loss_dict)
            
        loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = True)
        if summary_writer is not None:
            summary_writer.write_summary(loss_dict, train_flag = True)

        logger.info("TRAIN - AE_total: {:.3e}.".format(loss_dict["AE_total"].train))
        
        if misc.is_eval_epoch(epoch, cfg.SOLVER.MAX_EPOCH, cfg.TRAIN.EVAL_PERIOD):
            # Validation
            EpochAveMeter = AverageMeters(loss_name_list)
            rmse_all_sum = 0
            rmse_wet_sum = 0
            f1_metric_sum = 0

            VPTR_Enc.eval()
            VPTR_Dec.eval()
            VPTR_Disc.eval()

            save_dir=os.path.join(cfg.SAVE_RESULTS_PATH, "val_epoch{}".format(epoch))
            if not os.path.isdir(save_dir) and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
                os.mkdir(save_dir)

            for iter, sample in enumerate(val_loader, 0):
                iter_loss_dict, rmse_all, rmse_wet, f1_metric = val_single_iter(
                    VPTR_Enc, 
                    VPTR_Dec, 
                    VPTR_Disc,
                    sample[1], 
                    device, 
                    rmse_metric,
                    metrics_classif,
                    gan_loss,
                    mse_loss,
                    gdl_loss,
                    l1_loss,
                    cfg=cfg,
                    preprocessing_map=preprocessing_map,
                    print_image=print_image,
                    cur_iter=iter * word_size + rank,
                    save_dir=save_dir,
                )
                if cfg.NUM_GPUS > 1:
                    for key in iter_loss_dict.keys():
                        iter_loss_dict[key] = du.all_reduce([iter_loss_dict[key]])[0].item()
                    rmse_all, rmse_wet, f1_metric = du.all_reduce([rmse_all, rmse_wet, f1_metric])
                EpochAveMeter.iter_update(iter_loss_dict)
                rmse_all_sum += rmse_all
                rmse_wet_sum += rmse_wet
                f1_metric_sum += f1_metric

            mean_rmse_all = rmse_all_sum / (iter + 1)
            mean_rmse_wet = rmse_wet_sum / (iter + 1)
            mean_f1_metric = f1_metric_sum / (iter + 1)
            loss_dict = EpochAveMeter.epoch_update(loss_dict, epoch, train_flag = False)
            logger.info("VAL - AE_total: {:.3e}.".format(loss_dict["AE_total"].val))
            logger.info("VAL - RMSE all: {:.4f} m; RMSE wet: {:.4f} m; F1: {:.3f}".format(mean_rmse_all, mean_rmse_wet, mean_f1_metric))
            if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
                if summary_writer is not None:
                    summary_writer.add_scalars({"val_RMSE_all": mean_rmse_all, "val_RMSE_wet": mean_rmse_wet, "val_F1": mean_f1_metric}, epoch)
                    summary_writer.write_summary(loss_dict, train_flag = False)

                with open(os.path.join(save_dir, "rmse_stats.txt"), 'a') as f:
                    f.write("\nVAL: mean RMSE all maps: {:.4f} m\tmean RMSE wet cells: {:.4f} m; mean F1: {:.3f}".format(mean_rmse_all, mean_rmse_wet, mean_f1_metric))

        if is_checkpoint_epoch(epoch, cfg.SOLVER.MAX_EPOCH, cfg.TRAIN.CHECKPOINT_PERIOD, start_epoch) and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            save_ckpt(
                {'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec, 'VPTR_Disc': VPTR_Disc},
                {'optimizer_G': optimizer_G, 'optimizer_D': optimizer_D},
                epoch,
                cfg.CHECKPOINT_PATH
            )

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
    # --------------

    if start_epoch < cfg.SOLVER.MAX_EPOCH:
        train_time = datetime.now() - train_start
        eta_sec = int(train_time.total_seconds())
        logger.info("End of training. Duration: {}. Max_gpu_mem: {:.2f}/{:.2f}G.".format(timedelta(seconds=eta_sec), misc.gpu_mem_usage()[0], misc.gpu_mem_usage()[1]))

    ##################### Final AE test ###########################
    if cfg.TEST.ENABLE:
        logger.info(" ")
        logger.info("--- Run final AE test ---")
        # Init test dataset
        test_loader = loader.construct_loader(cfg, "test")
        preprocessing_map = test_loader.dataset.get_preprocessing()

        save_dir = os.path.join(cfg.SAVE_RESULTS_PATH, "test_epoch{}_FINAL".format(epoch))
        if not os.path.isdir(save_dir) and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            os.mkdir(save_dir)
        clas_metrics_app = None
        rmse_all_sum = 0
        rmse_wet_sum = 0
        VPTR_Enc = VPTR_Enc.eval()
        VPTR_Dec = VPTR_Dec.eval()
        test_start = datetime.now()

        logger.info("Start test loop")
        for iter, sample in enumerate(test_loader):
            rmse_all, rmse_wet, clas_metrics = test_show_samples_AE(
                VPTR_Enc, 
                VPTR_Dec, 
                sample[1], 
                save_dir, 
                preprocessing_map, 
                print_image,
                device, 
                rmse_metric,
                metrics_classif,
                cur_iter=iter * word_size + rank, 
                batch_size=test_loader.batch_size,
            )
            if cfg.NUM_GPUS > 1:
                rmse_all, rmse_wet, clas_metrics = du.all_reduce([rmse_all, rmse_wet, clas_metrics])
            rmse_all_sum += rmse_all
            rmse_wet_sum += rmse_wet
            if clas_metrics_app is None:
                clas_metrics_app = clas_metrics.unsqueeze(0)
            else:
                clas_metrics_app = torch.cat((clas_metrics_app, clas_metrics.unsqueeze(0)))
            #logger.info("FINAL TEST - iter {}: RMSE all: {:.4f} m; RMSE wet {:.4f} m; Precision: {:.3f}; Recall: {:.3f}; F1: {:.3f}".format(iter, rmse_all, rmse_wet, clas_metrics[0], clas_metrics[1], clas_metrics[2]))

        clas_metrics_app = torch.mean(clas_metrics_app, dim=0)
        test_time = datetime.now() - test_start

        if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            with open(os.path.join(save_dir, "rmse_stats.txt"), 'a') as f:
                f.write('\nMean RMSE all: {:.4f} m; mean RMSE wet: {:.4f} m; Mean precision: {:.3f}; Mean recall: {:.3f}; Mean F1: {:.3f}'.format(rmse_all_sum / (iter + 1), rmse_wet_sum / (iter + 1), clas_metrics_app[0], clas_metrics_app[1], clas_metrics_app[2]))

            logger.info(
                "End of final test - Duration: {}. Mean RMSE all: {:.4f} m; mean RMSE wet: {:.4f} m; Mean F1: {:.3f}; Max_gpu_mem: {:.2f}/{:.2f}G."
                .format(
                    timedelta(seconds=int(test_time.total_seconds())), 
                    rmse_all_sum / (iter + 1), 
                    rmse_wet_sum / (iter + 1), 
                    clas_metrics_app[2], 
                    misc.gpu_mem_usage()[0], 
                    misc.gpu_mem_usage()[1]
                )
            )

    if summary_writer is not None:
        summary_writer.close()