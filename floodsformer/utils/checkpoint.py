
""" Functions that handle saving and loading of checkpoints. """

import os
import torch
from fvcore.common.file_io import PathManager
from collections import OrderedDict

import floodsformer.utils.distributed as du
import floodsformer.utils.logg as logg
import torch.nn.functional as F
from pathlib import Path

logger = logg.get_logger(__name__)

def save_ckpt(Modules_dict, Optimizers_dict, epoch, save_dir, num_input_frames=None):
    '''
    Save model checkpoint.
    Args:
        Modules_dict: models to save.
        Optimizers_dict: optimizers to save.
        epoch (int): current epoch.
        save_dir (string): path to the folder in which save the checkpoint.
        num_input_frames (int): number of input frames used during the FS training.
    '''
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True) 
    ckpt_file = Path(save_dir).joinpath(f"epoch_{epoch}.tar")

    module_state_dict = {}
    for k, m in Modules_dict.items():
        try:
            module_state_dict[k] = m.module.state_dict()  # for compatibility in case of multi GPUs
        except:
            module_state_dict[k] = m.state_dict()
    optim_state_dict = {}
    for k, m in Optimizers_dict.items():
        optim_state_dict[k] = m.state_dict()

    if num_input_frames is None:
        torch.save({
            'epoch': epoch,
            'Module_state_dict': module_state_dict,
            'optimizer_state_dict': optim_state_dict,
        }, ckpt_file.absolute().as_posix())
    else:
        torch.save({
            'epoch': epoch,
            'num_input_frames': num_input_frames,
            'Module_state_dict': module_state_dict,
            'optimizer_state_dict': optim_state_dict,
        }, ckpt_file.absolute().as_posix())

def load_ckpt(ckpt_file, map_location=None):
    '''
    Load model checkpoint.
    Args:
        ckpt_file (string): path to the saved checkpoint.
        map_location (torch.device): the current device.
    '''
    ckpt = torch.load(ckpt_file, map_location=map_location)

    epoch = ckpt["epoch"]
    Modules_state_dict = ckpt['Module_state_dict']
    if 'optimizer_state_dict' in ckpt:
        Optimizers_state_dict = ckpt['optimizer_state_dict']
    else:
        Optimizers_state_dict = None

    # Number of input frames of the Transformer module
    if 'num_input_frames' in ckpt:
        num_input_frames = ckpt['num_input_frames']
    else:
        num_input_frames = None

    return Modules_state_dict, Optimizers_state_dict, epoch, num_input_frames

def resume_training_AE(module_dict, optimizer_dict, resume_ckpt, map_location=None):
    '''
    Resume the autoencoder pretrained weights.
    Args:
        module_dict: models.
        optimizer_dict: optimizer.
        resume_ckpt (string): path to the saved checkpoint.
        map_location (torch.device): the current device.
    '''
    logger.info("Loading AE preatrained model from: {}".format(resume_ckpt))
    modules_state_dict, optimizers_state_dict, start_epoch, _ = load_ckpt(resume_ckpt, map_location)
    #load_opt = True
    for k, m in module_dict.items():
        state_dict_load = modules_state_dict[k]
        logger.info("Loading {}:".format(k))
        try:
            msg = m.load_state_dict(state_dict_load, strict=False)
        except:
            interp_mode = 'nearest'  # Algorithm used for upsampling: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            #load_opt = False
            if m.n_downsampling == 4 and m.feat_dim == 528:
                msg = resumeAE_ds4_dim528(state_dict_load, m, interp_mode)
            elif m.n_downsampling == 4 and m.feat_dim == 512:
                msg = resumeAE_ds4_dim512(state_dict_load, m, interp_mode)
            elif m.n_downsampling == 5 and m.feat_dim == 768:
                msg = resumeAE_ds5_dim768(state_dict_load, m, interp_mode)
            else:
                raise RuntimeError("Unable to load the {} pretrained model.".format(k))
        logger.info(msg)

    ## Optimizer
    #if optimizers_state_dict is not None and load_opt:
    #    for k, m in optimizer_dict.items():
    #        state_dict = optimizers_state_dict[k]
    #        logger.info('Loading {}.'.format(k))
    #        m.load_state_dict(state_dict)

    return start_epoch

def resume_train_transformer(
        module_dict={}, 
        optimizer_dict={},
        resume_ckpt='',
        map_location=None,
        gpu_correction=1,
    ):
    '''
    Resume the transformer pretrained weights.
    Args:
        module_dict: transformer model.
        optimizer_dict: optimizer.
        resume_ckpt (string): path to the saved checkpoint.
        map_location (torch.device): the current device.
        gpu_correction (int): if cfg.NUM_GPUS>1 used to fix the name of modules when 
                              loading the checkpoint at the end of the training.
    '''
    logger.info("Loading Transformer pretrained model from: {}".format(resume_ckpt))
    modules_state_dict, optimizers_state_dict, start_epoch, num_input_frames = load_ckpt(resume_ckpt, map_location)
    load_opt = True
    for k, m in module_dict.items():
        if k not in modules_state_dict:
            logger.warning("Module {} not loaded!".format(k))
            continue
        state_dict = modules_state_dict[k]
        logger.info("Loading {}:".format(k))

        if gpu_correction > 1:
            # Fix the name of modules when loading the checkpoint at the end of the training.
            # Add 'module.' at the beginning of each name.
            for key in list(state_dict):
                new_key = 'module.' + key
                state_dict[new_key] = state_dict.pop(key)
            msg = m.load_state_dict(state_dict, strict=False)
            # Check missing or unexpected keys
            if len(msg[0]) != 0 or len(msg[1]) != 0:
                load_opt = False
            logger.info(msg)
            continue

        try:
            msg = m.load_state_dict(state_dict, strict=False)
        except:
            interp_mode = 'nearest'  # Algorithm used for upsampling: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            ## Check input size
            # Retrive the pretrained input size
            encH = m.transformer.H
            encW = m.transformer.W
            if 'frame_queries' in state_dict:
                old_encH = state_dict['frame_queries'].size(1)
                old_encW = state_dict['frame_queries'].size(2)
            elif 'transformer.encoder.layers.0.SpatialFFN.norm2.weight' in state_dict:
                old_encH = state_dict['transformer.encoder.layers.0.SpatialFFN.norm2.weight'].size(1)
                old_encW = state_dict['transformer.encoder.layers.0.SpatialFFN.norm2.weight'].size(2)
            else:
                old_encH = encH
                old_encW = encW
                logger.warning('Unable to find the pretrained input size. There may be errors in loading weights.')

            if encH != old_encH or encW != old_encW:
                logger.info('Resizing input size. encH from {} to {} and encW from {} to {}.'.format(old_encH, encH, old_encW, encW))
                load_opt = False
                new_state_dict = state_dict.copy()
                elem = None
                for key in state_dict:
                    if 'frame_queries' in key:
                        elem = state_dict[key].permute(0,3,1,2)
                        elem = F.interpolate(elem, size=(encH, encW), mode=interp_mode)
                        elem = elem.permute(0,2,3,1)
                        state_dict[key] = elem
                    elif 'SpatialFFN' in key and 'norm' in key:
                        elem = state_dict[key].unsqueeze(0)
                        elem = F.interpolate(elem, size=(encH, encW), mode=interp_mode)
                        state_dict[key] = elem[0,:,:,:]

            ## Check patch_size (window_size)
            # Retrive the pretrained patch size
            patch_size = m.window_size
            if 'lw_pos' in state_dict:
                old_patch_size = state_dict['lw_pos'].size(0)
            elif 'Tlw_pos' in state_dict:
                old_patch_size = state_dict['Tlw_pos'].size(1)
            else:
                old_patch_size = patch_size
                logger.warning('Unable to find the pretrained patch_size (window_size). There may be errors in loading weights.')

            if patch_size != old_patch_size:
                logger.info('Resizing patch_size (window_size) from {} to {}.'.format(old_patch_size, patch_size))
                load_opt = False
                new_state_dict = state_dict.copy()
                elem = None
                for key in state_dict:
                    if 'SLMHSA.attn.relative_position_bias_table' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=((2*patch_size-1)*(2*patch_size-1), m.nhead), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:,:]
                    elif 'SLMHSA.attn.relative_position_index' in key:
                        del new_state_dict[key]
                        #elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        #elem = F.interpolate(elem, size=(patch_size**2, patch_size**2), mode=interp_mode)
                        #new_state_dict[key] = elem[0,0,:,:]
                    elif 'Tlw_pos' in key:
                        elem = state_dict[key].permute(0,3,1,2)
                        elem = F.interpolate(elem, size=(patch_size, patch_size), mode=interp_mode)
                        elem = elem.permute(0,2,3,1)
                        new_state_dict[key] = elem
                    elif 'lw_pos' in key:
                        elem = state_dict[key].permute(2,0,1).unsqueeze(0)
                        elem = F.interpolate(elem, size=(patch_size, patch_size), mode=interp_mode)
                        elem = elem.permute(0,2,3,1)
                        new_state_dict[key] = elem[0,:,:,:]

                state_dict = new_state_dict
                del new_state_dict, elem, key

            ## Check number of frames in the input sequence
            # Retrive the pretrained past and future frames (in FS implementation corresponds to I+1)
            past_frames = m.num_past_frames   # corresponds to I
            future_frames = m.num_future_frames  # equal to 1
            if num_input_frames is not None:
                old_past_frames = num_input_frames
                old_future_frames = 1
                correction_t = past_frames != old_past_frames or future_frames != old_future_frames
            elif 'temporal_pos' in state_dict and 'frame_queries' in state_dict:
                old_past_frames = state_dict['frame_queries'].size(0)
                old_future_frames = state_dict['temporal_pos'].size(0) - old_past_frames
                correction_t = past_frames != old_past_frames or future_frames != old_future_frames
            elif 'Tlw_pos' in state_dict and 'frame_queries' in state_dict:
                old_past_frames = state_dict['frame_queries'].size(0)
                old_future_frames = state_dict['Tlw_pos'].size(0) - old_past_frames
                correction_t = past_frames != old_past_frames or future_frames != old_future_frames
            elif 'temporal_pos' in state_dict:
                correction_t = (past_frames + future_frames) != state_dict['temporal_pos'].size(0)
                if correction_t:
                    #logger.warning('Find only the sum of past and future frames (input sequence). Assume they are divided equally.')
                    old_past_frames = state_dict['temporal_pos'].size(0) // 2
                    old_future_frames = state_dict['temporal_pos'].size(0) - old_past_frames
            else:
                correction_t = False
                logger.warning('Unable to find sequence length of the pretrained model. There may be errors in loading weights.')

            if correction_t:
                logger.info('Resizing temporal informations. Number of frames in the sequence (I+1): {} -> {}.'.format(old_past_frames + old_future_frames, past_frames + future_frames))
                load_opt = False
                new_state_dict = state_dict.copy()
                elem = None
                for key in state_dict:
                    if 'frame_queries' in key:
                        elem = state_dict[key].transpose(0,3)
                        elem = F.interpolate(elem, size=(elem.size(2), future_frames), mode=interp_mode)
                        elem = elem.transpose(0,3)
                        new_state_dict[key] = elem
                    elif 'temporal_pos' in key:
                        elem = state_dict[key].transpose(0,1).unsqueeze(0)
                        elem = F.interpolate(elem, size=(past_frames+future_frames), mode=interp_mode)
                        elem = elem.transpose(1,2)
                        new_state_dict[key] = elem[0,:,:]
                    elif 'Tlw_pos' in key:
                        elem = state_dict[key].transpose(0,3)
                        elem = F.interpolate(elem, size=(elem.size(2), past_frames+future_frames), mode=interp_mode)
                        elem = elem.transpose(0,3)
                        new_state_dict[key] = elem

                state_dict = new_state_dict
                del new_state_dict, elem, key

            ## Check d_model
            # Retrive the pretrained d_model
            d_model = m.d_model
            if 'temporal_pos' in state_dict:
                old_d_model = state_dict['temporal_pos'].size(1)
            elif 'lw_pos' in state_dict:
                old_d_model = state_dict['lw_pos'].size(2)
            elif 'transformer.encoder.layers.0.norm1.weight' in state_dict:
                old_d_model = state_dict['transformer.encoder.layers.0.norm1.weight'].size(0)
            elif 'transformer.encoder.layers.0.norm4.weight' in state_dict:
                old_d_model = state_dict['transformer.encoder.layers.0.norm4.weight'].size(0)
            else:
                old_d_model = d_model
                logger.warning('Unable to find the pretrained d_model. There may be errors in loading weights.')

            if d_model != old_d_model:
                logger.info('Resizing d_model from {} to {}.'.format(old_d_model, d_model))
                new_state_dict = state_dict.copy()
                load_opt = False
                elem = None
                for key in state_dict:
                    if 'temporal_pos' in key:
                        elem = state_dict[key].unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model), mode=interp_mode)
                        new_state_dict[key] = elem[0,:,:]
                    elif 'lw_pos' in key:
                        elem = state_dict[key]
                        elem = F.interpolate(elem, size=(d_model), mode=interp_mode)
                        new_state_dict[key] = elem
                    elif ('SLMHSA.attn' in key and '.weight' in key) or 'temporal_MHSA.out_proj.weight' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model, d_model), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:,:]
                    elif 'SLMHSA.attn' in key and '.bias' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:]
                    elif '.SpatialFFN.' in key:
                        hidden_ratio = m.Spatial_FFN_hidden_ratio
                        if '.norm1.' in key or '.norm2.' in key:
                            elem = state_dict[key].transpose(0,2)
                            elem = F.interpolate(elem, size=(d_model * hidden_ratio), mode=interp_mode)
                            new_state_dict[key] = elem.transpose(0,2)
                        elif '.norm3.' in key:
                            elem = state_dict[key].transpose(0,2)
                            elem = F.interpolate(elem, size=(d_model), mode=interp_mode)
                            new_state_dict[key] = elem.transpose(0,2)
                        elif 'fc1.bias' in key or 'dw3x3.bias' in key:
                            elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                            elem = F.interpolate(elem, size=(d_model * hidden_ratio), mode=interp_mode)
                            new_state_dict[key] = elem[0,0,:]
                        elif 'fc2.bias' in key:
                            elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                            elem = F.interpolate(elem, size=(d_model), mode=interp_mode)
                            new_state_dict[key] = elem[0,0,:]
                        elif 'fc1.weight' in key:
                            elem = state_dict[key].permute(2,3,0,1)
                            elem = F.interpolate(elem, size=(d_model * hidden_ratio, d_model), mode=interp_mode)
                            new_state_dict[key] = elem.permute(2,3,0,1)
                        elif 'fc2.weight' in key:
                            elem = state_dict[key].permute(2,3,0,1)
                            elem = F.interpolate(elem, size=(d_model, d_model * hidden_ratio), mode=interp_mode)
                            new_state_dict[key] = elem.permute(2,3,0,1)
                        elif 'dw3x3.weight' in key:
                            elem = state_dict[key].transpose(0,3)
                            elem = F.interpolate(elem, size=(elem.size(2), d_model * hidden_ratio), mode=interp_mode)
                            new_state_dict[key] = elem.transpose(0,3)
                    elif '.norm' in key or 'temporal_MHSA.out_proj.bias' in key or '.linear2.bias' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:]
                    elif 'temporal_MHSA.in_proj_weight' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model * 3, d_model), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:,:]
                    elif 'temporal_MHSA.in_proj_bias' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model * 3), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:]
                    elif '.linear1.weight' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model * m.Spatial_FFN_hidden_ratio, d_model), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:,:]
                    elif '.linear2.weight' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model, d_model * m.Spatial_FFN_hidden_ratio), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:,:]
                    elif '.linear1.bias' in key:
                        elem = state_dict[key].unsqueeze(0).unsqueeze(0)
                        elem = F.interpolate(elem, size=(d_model * m.Spatial_FFN_hidden_ratio), mode=interp_mode)
                        new_state_dict[key] = elem[0,0,:]
                    
                state_dict = new_state_dict
                del new_state_dict, elem, key

            ## Check if qkv weights of SLMHSA are in the same tensor or not.
            if 'transformer.encoder.layers.0.SLMHSA.attn.k_proj.weight' in state_dict and 'transformer.encoder.layers.0.SLMHSA.attn.in_proj_weight' in m.state_dict():
                # Change from q, k, v to qkv
                logger.info('Changing SLHMSA.attn weights: join q_proj, k_proj, v_proj to obtain in_proj.')
                load_opt = False
                new_state_dict = state_dict.copy()
                for key in state_dict:
                    if 'SLMHSA.attn.q_proj.' in key:
                        key_q = key
                        key_k = key.replace('.q_proj.','.k_proj.')
                        key_v = key.replace('.q_proj.','.v_proj.')

                        if 'weight' in key:
                            new_key = key.replace('.q_proj.weight','.in_proj_weight')
                        elif 'bias' in key:
                            new_key = key.replace('.q_proj.bias','.in_proj_bias')
                        else:
                            logger.warning("Error in loading SLMHSA weights.")

                        new_state_dict[new_key] = torch.cat(
                            (state_dict[key_q], state_dict[key_k], state_dict[key_v]),
                            dim=0
                        )

                        del new_state_dict[key_q], new_state_dict[key_k], new_state_dict[key_v], key_q, key_k, key_v, new_key

                state_dict = new_state_dict
                del key, new_state_dict

            elif 'transformer.encoder.layers.0.SLMHSA.attn.k_proj.weight' in m.state_dict() and 'transformer.encoder.layers.0.SLMHSA.attn.in_proj_weight' in state_dict:
                # Change from qkv to q, k, v
                logger.info('Changing SLHMSA.attn weights: split in_proj to obtain q_proj, k_proj, v_proj.')
                load_opt = False
                new_state_dict = state_dict.copy()
                for key in state_dict:
                    if 'SLMHSA.attn.in_proj' in key:  # in_proj_weight
                        key_q = key.replace('in_proj_','q_proj.')
                        key_k = key.replace('in_proj_','k_proj.')
                        key_v = key.replace('in_proj_','v_proj.')

                        q_proj, k_proj, v_proj = torch.split(state_dict[key], state_dict[key].shape[0] // 3, dim=0)

                        new_state_dict[key_q] = q_proj
                        new_state_dict[key_k] = k_proj
                        new_state_dict[key_v] = v_proj

                        del new_state_dict[key], key_q, key_k, key_v, q_proj, k_proj, v_proj

                state_dict = new_state_dict
                del key, new_state_dict

            ## Loading the weights
            msg = m.load_state_dict(state_dict, strict=False)

        # Check missing or unexpected keys
        if len(msg[0]) != 0 or len(msg[1]) != 0:
            load_opt = False
        logger.info(msg)

    ## Optimizer
    if optimizers_state_dict is not None and load_opt:
        for k, m in optimizer_dict.items():
            state_dict = optimizers_state_dict[k]
            logger.info('Loading {}'.format(k))
            m.load_state_dict(state_dict)
    else:
        logger.info('Optimizer not loaded')

    return start_epoch, num_input_frames

def make_checkpoint_dir(cfg):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        cfg (CfgNode): configs. Details can be found in: floodsformer/config/defaults.py
    """
    checkpoint_dir = cfg.CHECKPOINT_PATH
    # Create the checkpoint dir from the master process
    if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS) and not PathManager.exists(checkpoint_dir):
        try:
            PathManager.mkdirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir

def is_checkpoint_epoch(cur_epoch, max_epoch, checkpoint_period, start_epoch):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cur_epoch (int): current number of epoch of the model.
        max_epoch (int): last epoch.
        checkpoint_period (int): save model weights every checkpoint_period epochs.
        start_epoch (int): the first epoch of the training.
    """
    if cur_epoch + 1 == max_epoch:
        return True
    return cur_epoch % checkpoint_period == 0 and start_epoch != cur_epoch

def resumeAE_ds4_dim528(state_dict_load, m, interp_mode):
    """ Used if Autoencoder downsampling == 4 and feat_dim == 528 """
    logger.info("Changing some parameter dimensions.")
    new_state_dict = OrderedDict()
    for key in state_dict_load:
        # VPTR_enc
        if "encoder.model." in key and ".conv_block." in key:
            # Increase the layer number of 3
            num = int(key[14:16]) + 3
            new_key = key.replace(key[14:16], str(num))
            new_state_dict[new_key] = state_dict_load[key]
        elif "encoder.model.10.weight" in key:
            elem = state_dict_load[key].permute(2, 3, 0, 1)
            elem1 = F.interpolate(elem, size=(512, 256), mode=interp_mode)
            new_state_dict[key] = elem1.permute(2, 3, 0, 1)

            elem1 = F.interpolate(elem, size=(528, 512), mode=interp_mode)
            new_state_dict["encoder.model.13.weight"] = elem1.permute(2, 3, 0, 1)
        elif "encoder.model.11." in key and not "num_batches_tracked" in key:
            elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
            elem = F.interpolate(elem, size=(512), mode=interp_mode)
            new_state_dict[key] = elem[0,0,:]

            new_key = key.replace('.11.', '.14.')
            new_state_dict[new_key] = state_dict_load[key]
        elif "encoder.model.11.num_batches_tracked" in key:
            new_key = key.replace('.11.', '.14.')
            new_state_dict[key] = state_dict_load[key]
            new_state_dict[new_key] = state_dict_load[key]

        # VPTR_dec
        elif "decoder.model.0.weight" in key:
            elem = state_dict_load[key].permute(2, 3, 0, 1)
            elem1 = F.interpolate(elem, size=(528, 512), mode=interp_mode)
            new_state_dict[key] = elem1.permute(2, 3, 0, 1)

            elem1 = F.interpolate(elem, size=(512, 256), mode=interp_mode)
            new_state_dict["decoder.model.3.weight"] = elem1.permute(2, 3, 0, 1)
        elif "decoder.model.1." in key and not "num_batches_tracked" in key:
            elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
            elem = F.interpolate(elem, size=(512), mode=interp_mode)
            new_state_dict[key] = elem[0,0,:]

            new_key = key.replace('.1.', '.4.')
            new_state_dict[new_key] = state_dict_load[key]
        elif "decoder.model.1.num_batches_traked" in key:
            new_key = key.replace('.1.', '.4.')
            new_state_dict[key] = state_dict_load[key]
            new_state_dict[new_key] = state_dict_load[key]
        elif "decoder.model." in key:
            # Increase the layer number of 3
            if '.10.' in key:
                new_key = key.replace('.10.', '.13.')
                new_state_dict[new_key] = state_dict_load[key]
                continue
            num = int(key[14]) + 3
            new_key = key.replace(key[14], str(num))
            new_state_dict[new_key] = state_dict_load[key]
        else:
            new_state_dict[key] = state_dict_load[key]

    ## Loading the weights
    msg = m.load_state_dict(new_state_dict, strict=False)
    return msg

def resumeAE_ds4_dim512(state_dict_load, m, interp_mode):
    """ Used if Autoencoder downsampling == 4 and feat_dim == 512 """
    logger.info("Changing some parameter dimensions.")
    new_state_dict = OrderedDict()

    for key in state_dict_load:
        # VPTR_enc
        if "num_batches_tracked" in key:
            continue
        elif "encoder.model." in key:
            if "encoder.model.1." in key:
                elem = state_dict_load[key].transpose(0,3)
                elem = F.interpolate(elem, size=(7, 32), mode=interp_mode)
                new_state_dict[key] = elem.transpose(0,3)
            elif "encoder.model.2." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.2.', '.5.')
                new_state_dict[new_key] = elem

                elem = elem.unsqueeze(0).unsqueeze(0)
                elem = F.interpolate(elem, size=(32), mode=interp_mode)
                new_state_dict[key] = elem[0,0,:]
            elif "encoder.model.4." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.4.', '.7.')
                new_state_dict[new_key] = elem

                elem = elem.permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(64, 32), mode=interp_mode)
                new_state_dict[key] = elem.permute(2, 3, 0, 1)
            elif "encoder.model.5." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.5.', '.8.')
                new_state_dict[new_key] = elem
            elif "encoder.model.7." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.7.', '.10.')
                new_state_dict[new_key] = elem
            elif "encoder.model.8." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.8.', '.11.')
                new_state_dict[new_key] = elem
            elif "encoder.model.10." in key:
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                new_key = key.replace('.10.', '.13.')
                elem1 = F.interpolate(elem, size=(512, 256), mode=interp_mode)
                new_state_dict[new_key] = elem1.permute(2, 3, 0, 1)
            elif "encoder.model.11." in key:
                elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
                new_key = key.replace('.11.', '.14.')
                elem1 = F.interpolate(elem, size=(512), mode=interp_mode)
                new_state_dict[new_key] = elem1[0,0,:]

            # ResnetBlock
            elif 'conv_block.1.' in key or 'conv_block.5.' in key:
                num = int(key[14:16]) + 3
                new_key = key.replace(key[14:16], str(num))
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(512, 512), mode=interp_mode)
                new_state_dict[new_key] = elem.permute(2, 3, 0, 1)
            elif 'conv_block.2.' in key or 'conv_block.6.' in key:
                num = int(key[14:16]) + 3
                new_key = key.replace(key[14:16], str(num))
                elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
                elem = F.interpolate(elem, size=(512), mode=interp_mode)
                new_state_dict[new_key] = elem[0,0,:]
            else:
                new_state_dict[key] = state_dict_load[key]

        # VPTR_dec
        elif "decoder.model." in key:
            if "decoder.model.0." in key:
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(512, 256), mode=interp_mode)
                new_state_dict[key] = elem.permute(2, 3, 0, 1)

            elif "decoder.model.6." in key:
                new_state_dict[key] = state_dict_load[key]

                new_key = key.replace('.6.', '.9.')
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(64, 32), mode=interp_mode)
                new_state_dict[new_key] = elem.permute(2, 3, 0, 1)

            elif "decoder.model.7." in key:
                new_state_dict[key] = state_dict_load[key]

                new_key = key.replace('.7.', '.10.')
                elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
                elem = F.interpolate(elem, size=(32), mode=interp_mode)
                new_state_dict[new_key] = elem[0,0,:]
            elif "decoder.model.10.weight" in key:
                new_key = key.replace('.10.', '.13.')
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(1, 32), mode=interp_mode)
                new_state_dict[new_key] = elem.permute(2, 3, 0, 1)
            elif "decoder.model.10.bias" in key:
                new_key = key.replace('.10.', '.13.')
                new_state_dict[new_key] = state_dict_load[key]
            else:
                new_state_dict[key] = state_dict_load[key]
        else:
            new_state_dict[key] = state_dict_load[key]
    
    ## Loading the weights
    msg = m.load_state_dict(new_state_dict, strict=False)
    return msg


def resumeAE_ds5_dim768(state_dict_load, m, interp_mode):
    """ Used if Autoencoder downsampling == 5 and feat_dim == 768 """
    logger.info("Changing some parameter dimensions.")
    new_state_dict = OrderedDict()

    for key in state_dict_load:
        # VPTR_enc
        if "num_batches_tracked" in key:
            continue
        elif "encoder.model." in key:
            if "encoder.model.1." in key:
                elem = state_dict_load[key].transpose(0,3)
                elem = F.interpolate(elem, size=(7, 32), mode=interp_mode)
                new_state_dict[key] = elem.transpose(0,3)
            elif "encoder.model.2." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.2.', '.5.')
                new_state_dict[new_key] = elem

                elem = elem.unsqueeze(0).unsqueeze(0)
                elem = F.interpolate(elem, size=(32), mode=interp_mode)
                new_state_dict[key] = elem[0,0,:]
            elif "encoder.model.4." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.4.', '.7.')
                new_state_dict[new_key] = elem

                elem = elem.permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(64, 32), mode=interp_mode)
                new_state_dict[key] = elem.permute(2, 3, 0, 1)
            elif "encoder.model.5." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.5.', '.8.')
                new_state_dict[new_key] = elem
            elif "encoder.model.7." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.7.', '.10.')
                new_state_dict[new_key] = elem
            elif "encoder.model.8." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.8.', '.11.')
                new_state_dict[new_key] = elem
            elif "encoder.model.10." in key:
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                new_key = key.replace('.10.', '.13.')
                elem1 = F.interpolate(elem, size=(512, 256), mode=interp_mode)
                new_state_dict[new_key] = elem1.permute(2, 3, 0, 1)

                new_key = key.replace('.10.', '.16.')
                elem1 = F.interpolate(elem, size=(768, 512), mode=interp_mode)
                new_state_dict[new_key] = elem1.permute(2, 3, 0, 1)
            elif "encoder.model.11." in key:
                elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
                new_key = key.replace('.11.', '.14.')
                elem1 = F.interpolate(elem, size=(512), mode=interp_mode)
                new_state_dict[new_key] = elem1[0,0,:]

                new_key = key.replace('.11.', '.17.')
                elem1 = F.interpolate(elem, size=(768), mode=interp_mode)
                new_state_dict[new_key] = elem1[0,0,:]
            # ResnetBlock
            elif 'conv_block.1.' in key or 'conv_block.5.' in key:
                num = int(key[14:16]) + 6
                new_key = key.replace(key[14:16], str(num))
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(768, 768), mode=interp_mode)
                new_state_dict[new_key] = elem.permute(2, 3, 0, 1)
            elif 'conv_block.2.' in key or 'conv_block.6.' in key:
                num = int(key[14:16]) + 6
                new_key = key.replace(key[14:16], str(num))
                elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
                elem = F.interpolate(elem, size=(768), mode=interp_mode)
                new_state_dict[new_key] = elem[0,0,:]
            else:
                new_state_dict[key] = state_dict_load[key]

        # VPTR_dec
        elif "decoder.model." in key:
            if "decoder.model.0." in key:
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                elem1 = F.interpolate(elem, size=(768, 512), mode=interp_mode)
                new_state_dict[key] = elem1.permute(2, 3, 0, 1)

                new_key = key.replace('.0.', '.3.')
                elem1 = F.interpolate(elem, size=(512, 256), mode=interp_mode)
                new_state_dict[new_key] = elem1.permute(2, 3, 0, 1)
            elif "decoder.model.1." in key:
                elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
                new_key = key.replace('.1.', '.4.')
                new_state_dict[new_key] = elem[0,0,:]

                elem = F.interpolate(elem, size=(512), mode=interp_mode)
                new_state_dict[key] = elem[0,0,:]
            elif "decoder.model.3." in key:
                new_key = key.replace('.3.', '.6.')
                new_state_dict[new_key] = state_dict_load[key]
            elif "decoder.model.4." in key:
                new_key = key.replace('.4.', '.7.')
                new_state_dict[new_key] = state_dict_load[key]
            elif "decoder.model.6." in key:
                elem = state_dict_load[key]
                new_key = key.replace('.6.', '.9.')
                new_state_dict[new_key] = elem

                new_key = key.replace('.6.', '.12.')
                elem = elem.permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(64, 32), mode=interp_mode)
                new_state_dict[new_key] = elem.permute(2, 3, 0, 1)
            elif "decoder.model.7." in key:
                new_key = key.replace('.7.', '.10.')
                new_state_dict[new_key] = state_dict_load[key]

                new_key = key.replace('.7.', '.13.')
                elem = state_dict_load[key].unsqueeze(0).unsqueeze(0)
                elem = F.interpolate(elem, size=(32), mode=interp_mode)
                new_state_dict[new_key] = elem[0,0,:]
            elif "decoder.model.10.weight" in key:
                new_key = key.replace('.10.', '.16.')
                elem = state_dict_load[key].permute(2, 3, 0, 1)
                elem = F.interpolate(elem, size=(1, 32), mode=interp_mode)
                new_state_dict[new_key] = elem.permute(2, 3, 0, 1)
            elif "decoder.model.10.bias" in key:
                new_key = key.replace('.10.', '.16.')
                new_state_dict[new_key] = state_dict_load[key]
            else:
                new_state_dict[key] = state_dict_load[key]
        else:
            new_state_dict[key] = state_dict_load[key]

    ## Loading the weights
    msg = m.load_state_dict(new_state_dict, strict=False)
    return msg
