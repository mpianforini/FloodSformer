
""" Meters. """

class AverageMeters(object):
    def __init__(self, loss_name_list):
        self.loss_name_list = loss_name_list
        self.meters = {}
        for name in loss_name_list:
            self.meters[name] = BatchAverageMeter(name, ':.10e')
    
    def iter_update(self, iter_loss_dict):
        for k, v in iter_loss_dict.items():
            self.meters[k].update(v)
    
    def epoch_update(self, loss_dict, epoch, train_flag = True):
        del loss_dict['epochs']
        if train_flag:
            for k, v in loss_dict.items():
                try:
                    #v.train.append(self.meters[k].avg)
                    v.train = self.meters[k].avg
                except AttributeError:
                    pass
                except KeyError:
                    #v.train.append(0)
                    v.train = 0
        else:
            for k, v in loss_dict.items():
                try:
                    #v.val.append(self.meters[k].avg)
                    v.val = self.meters[k].avg
                except AttributeError:
                    pass
                except KeyError:
                    #v.val.append(0)
                    v.val = 0
        loss_dict['epochs'] = epoch

        return loss_dict
    
class BatchAverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L363
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)