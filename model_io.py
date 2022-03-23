# encoding=utf-8

import os
import torch


def save_model(opt, model, optimizer):
    """
    Save the models
    """
    save_path = os.path.join(opt.model_path, "checkpoint{}.tar".format(opt.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if opt.nodes > 1:
        torch.save(model.module.state_dict(), save_path)
    else:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
    print("{:s} saved.".format(save_path))
