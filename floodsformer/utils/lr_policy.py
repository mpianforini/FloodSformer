
""" Learning rate scheduler. """

import math

def set_lr(optimizers, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizers (list): list of the optimizers used to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

class lr_constant():
    def __init__(self, cfg):
        """
        Constant lr.
        Args:
            cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        """
        self.base_lr = cfg.SOLVER.BASE_LR
        self.warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
        self.warmup_start_lr = cfg.SOLVER.WARMUP_START_LR

    def __call__(self, cur_epoch, optimizers, summary_writer):
        """
        Args:
            cur_epoch (float): the number of epoch of the current training stage.
            optimizers (list): list of the optimizers used to optimize the current network.
            summary_writer: function to log information to Tensorboard.
        """
        if self.warmup_epochs > 0:
            # Perform warm up
            if cur_epoch <= self.warmup_epochs:
                alpha = (self.base_lr - self.warmup_start_lr) / self.warmup_epochs
                lr = cur_epoch * alpha + self.warmup_start_lr

                # Set the lr
                set_lr(optimizers, lr)
            else:
                lr =  self.base_lr

            # Log lr to Tensorboard
            if summary_writer is not None:
                summary_writer.add_scalars({"lr": lr}, cur_epoch)

            return lr
        else:
            return self.base_lr

class cosine_scheduler():
    def __init__(self, cfg):
        """
        Retrieve the learning rate to specified values at specified epoch with the cosine learning rate schedule.
        Details can be found in: Ilya Loshchilov, and  Frank Hutter, 2017. SGDR: Stochastic Gradient Descent With Warm Restarts.
        Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/utils/lr_policy.py
        Args:
            cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        """
        self.cosine_end_lr = cfg.SOLVER.COSINE_END_LR
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.base_lr = cfg.SOLVER.BASE_LR
        self.warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
        self.warmup_start_lr = cfg.SOLVER.WARMUP_START_LR

    def __call__(self, cur_epoch, optimizers, summary_writer):
        """
        Args:
            cur_epoch (float): the number of epoch of the current training stage.
            optimizers (list): list of the optimizers used to optimize the current network.
            summary_writer: function to log information to Tensorboard.
        """
        if cur_epoch < self.warmup_epochs:
            # Perform warm up
            lr_end = self.cosine_end_lr + (self.base_lr - self.cosine_end_lr) * (math.cos(math.pi * self.warmup_epochs / self.max_epoch) + 1.0) * 0.5
            alpha = (lr_end - self.warmup_start_lr) / self.warmup_epochs
            lr = cur_epoch * alpha + self.warmup_start_lr
        else:
            lr = self.cosine_end_lr + (self.base_lr - self.cosine_end_lr) * (math.cos(math.pi * cur_epoch / self.max_epoch) + 1.0) * 0.5

        # Set the lr
        set_lr(optimizers, lr)

        # Log lr to Tensorboard
        if summary_writer is not None:
            summary_writer.add_scalars({"lr": lr}, cur_epoch)

        return lr

class StepsWithRelativeLrs():
    def __init__(self, cfg):
        """
        Retrieve the learning rate to specified values at specified epoch with the
        steps with relative learning rate schedule.
        Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/utils/lr_policy.py
        Args:
            cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        """
        self.steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
        self.lrs = cfg.SOLVER.LRS
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
        self.warmup_start_lr = cfg.SOLVER.WARMUP_START_LR

    def __call__(self, cur_epoch, optimizers, summary_writer):
        """
        Args:
            cur_epoch (float): the number of epoch of the current training stage.
            optimizers (list): list of the optimizers used to optimize the current network.
            summary_writer: function to log information to Tensorboard.
        """
        if cur_epoch < self.warmup_epochs:
            # Perform warm up
            for ind, step in enumerate(self.steps):
                if self.warmup_epochs < step:
                    break
            lr_end = self.lrs[ind]
            alpha = (lr_end - self.warmup_start_lr) / self.warmup_epochs
            lr = cur_epoch * alpha + self.warmup_start_lr
        else:
            for ind, step in enumerate(self.steps):
                if cur_epoch < step:
                    break
            lr = self.lrs[ind]
        
        # Set the lr
        set_lr(optimizers, lr)

        # Log lr to Tensorboard
        if summary_writer is not None:
            summary_writer.add_scalars({"lr": lr}, cur_epoch)

        return lr

class CosAnnealWarmRest():
    def __init__(self, cfg):
        """
        Set the learning rate of each parameter group using a cosine annealing schedule with warm restarts.
        Details can be found in: Ilya Loshchilov, and  Frank Hutter, 2017. SGDR: Stochastic Gradient Descent With Warm Restarts.
        Args:
            cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        """
        self.cosine_end_lr = cfg.SOLVER.COSINE_END_LR
        self.base_lr = cfg.SOLVER.BASE_LR
        self.warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
        self.warmup_start_lr = cfg.SOLVER.WARMUP_START_LR
        self.restart_epoch = cfg.SOLVER.RESTART_EPOCH            # Number of iterations for the first restart
        self.restart_epoch_mult=cfg.SOLVER.RESTART_EPOCH_MULTIP  # A factor multiplies self.restart_epoch at each restart
        self.count = 0
        assert self.warmup_epochs <= self.restart_epoch, "Set a warmup epoch lower than the cosine annealing restart epoch"
        assert self.restart_epoch != 0

    def __call__(self, cur_epoch, optimizers, summary_writer):
        """
        Args:
            cur_epoch (float): the number of epoch of the current training stage.
            optimizers (list): list of the optimizers used to optimize the current network.
            summary_writer: function to log information to Tensorboard.
        """
        if cur_epoch < self.warmup_epochs:
            # Perform warm up
            lr_end = self.cosine_end_lr + (self.base_lr - self.cosine_end_lr) * (math.cos(math.pi * self.warmup_epochs / self.restart_epoch) + 1.0) * 0.5
            alpha = (lr_end - self.warmup_start_lr) / self.warmup_epochs
            lr = cur_epoch * alpha + self.warmup_start_lr
            self.count = self.count + 1
        else:
            if self.count >= self.restart_epoch:
                lr = self.base_lr
                self.restart_epoch = max(int(self.restart_epoch * self.restart_epoch_mult), 1)
                self.count = 1
            else:
                lr = self.cosine_end_lr + (self.base_lr - self.cosine_end_lr) * (math.cos(math.pi * self.count / self.restart_epoch) + 1.0) * 0.5
                self.count = self.count + 1

        # Set the lr
        set_lr(optimizers, lr)

        # Log lr to Tensorboard
        if summary_writer is not None:
            summary_writer.add_scalars({"lr": lr}, cur_epoch)

        return lr

class lr_scheduler():
    def __init__(self, cfg):
        """
        Learning rate scheduler.
        Args:
            cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        """
        print(cfg.SOLVER.LR_POLICY)
        # Select the correct lr policy
        if cfg.SOLVER.LR_POLICY == 'constant':
            self.lr_function = lr_constant(cfg)
        elif cfg.SOLVER.LR_POLICY == 'CosAnnealWR':
            self.lr_function = CosAnnealWarmRest(cfg)
        elif cfg.SOLVER.LR_POLICY == 'cosine':
            self.lr_function = cosine_scheduler(cfg)
        elif cfg.SOLVER.LR_POLICY == 'steps_with_relative_lrs':
            self.lr_function = StepsWithRelativeLrs(cfg)
        else:
            raise NotImplementedError("Unknown LR scheduler: {}".format(cfg.SOLVER.LR_POLICY))

    def __call__(self, cur_epoch, optimizers, summary_writer):
        """
        Args:
            cur_epoch (float): the number of epoch of the current training stage.
            optimizers (list): list of the optimizers used to optimize the current network.
            summary_writer: function to log information to Tensorboard.
        """
        return self.lr_function(cur_epoch, optimizers, summary_writer)


if __name__ == "__main__":
    """ Code to check the lr scheduler """
    import matplotlib.pyplot as plt
    import torch
    class get_solver():
        def __init__(self):
            self.MAX_EPOCH = 100
            self.LR_POLICY = 'CosAnnealWR'  # [constant, cosine, steps_with_relative_lrs, CosAnnealWR]
            self.BASE_LR = 1e-3
            self.COSINE_END_LR = 1e-5
            self.STEPS = [10,20,30]
            self.LRS = [1e-3, 5e-4, 1e-4, 5e-5]
            self.WARMUP_EPOCHS = 0
            self.WARMUP_START_LR = 1e-6
            self.RESTART_EPOCH = 25
            self.RESTART_EPOCH_MULTIP = 1.0

    class get_cfg():
        def __init__(self):
            self.SOLVER = get_solver()
            assert self.SOLVER.LR_POLICY in ['constant', 'cosine', 'steps_with_relative_lrs', 'CosAnnealWR'], "Learning rate policy ({}) not supported.".format(cfg.SOLVER.LR_POLICY)
            if self.SOLVER.LR_POLICY == "cosine": assert self.SOLVER.COSINE_END_LR < self.SOLVER.BASE_LR
            if self.SOLVER.LR_POLICY == "steps_with_relative_lrs": assert len(self.SOLVER.LRS) == len(self.SOLVER.STEPS) + 1

    cfg = get_cfg()
    lr = cfg.SOLVER.BASE_LR
    lr_append = []
    summary_writer = None

    model = torch.nn.Linear(3, 256)
    optimizers = []
    for i in range(1):  # set a number of optimizers
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = cfg.SOLVER.BASE_LR)
        optimizers.append(optimizer)

    lr_sched = lr_scheduler(cfg)
    for epoch in range(cfg.SOLVER.MAX_EPOCH + 1):
        lr = lr_sched(epoch, optimizers, summary_writer)

        lr_append.append(lr)
        #print(f"Epoch {epoch}: lr = {lr}")

    # plot the lr
    xpts = range(0, cfg.SOLVER.MAX_EPOCH + 1)
    plt.plot(xpts, lr_append)
    plt.xlabel('Epoch')
    plt.ylabel('Lr')
    plt.xticks(range(0, cfg.SOLVER.MAX_EPOCH + 1, 5))
    plt.title('Lr scheduler: {}'.format(cfg.SOLVER.LR_POLICY))
    plt.show()
