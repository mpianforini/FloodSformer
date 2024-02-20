
""" Tensorboard configurations. """

import os
from torch.utils.tensorboard import SummaryWriter
import floodsformer.utils.logg as logg

logger = logg.get_logger(__name__)

class TensorboardWriter(object):
    """ Helper class to log information to Tensorboard. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        """
        if cfg.TENSORBOARD.LOG_DIR == "":
            log_dir = os.path.join(cfg.OUTPUT_DIR, "runs-{}".format(cfg.DATA.DATASET))
        else:
            log_dir = os.path.join(cfg.TENSORBOARD.LOG_DIR)

        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info("To see logged results in Tensorboard, please launch using the command: `tensorboard --port=6006 --logdir {}`".format(log_dir))

    def add_scalars(self, data_dict, global_step=None):
        """
        Add multiple scalars to Tensorboard logs.
        Args:
            data_dict (dict): key is a string specifying the tag of value.
            global_step (Optinal[int]): Global step value to record.
        """
        if self.writer is not None:
            for key, item in data_dict.items():
                self.writer.add_scalar(key, item, global_step)

    def add_graph(self, model, input_to_model, verbose=False):
        """
        Add graph data to summary.
        Args:
            model (torch.nn.Module): model to draw.
            input_to_model (Tensor): a dummy input tensor.
            verbose (bool): print graph structure.
        """
        if self.writer is not None:
            self.writer.add_graph(model, input_to_model=input_to_model, verbose=verbose)

    def write_summary(self, in_loss_dict, train_flag = True):
        loss_dict = in_loss_dict.copy()
        epoch = loss_dict['epochs']
        del loss_dict['epochs']
        if train_flag:
            for k, v in loss_dict.items():
                self.writer.add_scalars(k, {'train': v.train}, epoch)
        else:
            for k, v in loss_dict.items():
                self.writer.add_scalars(k, {'val': v.val}, epoch)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()