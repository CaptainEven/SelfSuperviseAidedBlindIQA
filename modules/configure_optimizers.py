# encoding=utf-8
import os.path

import torch

from modules import scheduler as sch


def configure_optimizers(args, model, cur_iter=-1, load_optimizer=True):
    """
    Optimizer
    """
    iters = args.iters

    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.0,
            'layer_adaptation': False,
        },
    ]

    LR = args.lr

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
        )
    else:
        raise NotImplementedError

    if args.reload and load_optimizer:
        optimizer_path = args.model_path + 'optimizer.tar'
        if os.path.isfile(optimizer_path):
            fl = torch.load(optimizer_path)
            optimizer.load_state_dict(fl['optimizer'])
            cur_iter = fl['scheduler']['last_epoch'] - 1
            print("{:s} loaded.".format(optimizer_path))
        else:
            print("[Err]: invalid optimizer file path: {:s}".format(optimizer_path))
    else:
        print("[Info]: optimizer not loaded.")

    if args.lr_schedule == 'warmup-anneal':
        scheduler = sch.LinearWarmupAndCosineAnneal(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    elif args.lr_schedule == 'linear':
        scheduler = sch.LinearLR(optimizer, iters, last_epoch=cur_iter)
    elif args.lr_schedule == 'const':
        scheduler = sch.LinearWarmupAndConstant(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    else:
        raise NotImplementedError

    if args.reload and load_optimizer:
        scheduler.load_state_dict(fl['scheduler'])

    return optimizer, scheduler
