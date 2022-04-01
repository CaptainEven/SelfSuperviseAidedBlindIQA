# encoding=utf-8

import argparse
import math
import os

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset

from utils.utils import find_most_free_gpu, select_device


class RegressorDataset(Dataset):
    def __init__(self, feat_npy_path, score_npy_path):
        feat_npy_path = os.path.abspath(feat_npy_path)
        score_npy_path = os.path.abspath(score_npy_path)

        if not os.path.isfile(feat_npy_path):
            print("[Err]: invalid feature file path {:s}"
                  .format(feat_npy_path))
            exit(-1)
        if not os.path.isfile(score_npy_path):
            print("[Err]: invalid score file path {:s}"
                  .format(score_npy_path))
            exit(-1)

        self.features = np.load(feat_npy_path)
        self.scores = np.load(score_npy_path)
        self.n = self.features.shape[0]
        print("[Info]: total {:d} samples.".format(self.n))

    def __getitem__(self, idx):
        feature = np.array(self.features[idx], dtype=np.float32)
        score = np.array(self.scores[idx], np.float32)

        return torch.tensor(feature), torch.tensor(score)

    def __len__(self):
        return self.n


class RegressorPt(torch.nn.Module):
    def __init__(self, in_size, hidden_size=1024):
        """
        Init
        """
        super(RegressorPt, self).__init__()
        self.linear = torch.nn.Sequential(torch.nn.Linear(in_size, hidden_size),
                                          torch.nn.Linear(hidden_size, 1))

    def forward(self, x):
        """
        Forward pass
        """
        return self.linear.forward(x)


def run(opt):
    """
    Run the regressor training
    """
    ## ----- Set up device
    dev = str(find_most_free_gpu())
    print("[Info]: Using GPU {:s}.".format(dev))
    dev = select_device(dev)
    opt.device = dev
    dev = opt.device

    ## ----- Set up the network: mlp
    # net = torch.nn.Sequential(torch.nn.Linear(opt.input_size, 1024),
    #                           torch.nn.Linear(1024, 1))
    net = RegressorPt(opt.input_size, opt.hidden_size)
    net = net.to(dev)

    ## ---------- Load the checkpoint
    if opt.reload:
        if opt.ckpt_path != "":
            if not os.path.isfile(opt.ckpt_path):
                print("[Err]: invalid network weights path {:s}".format(opt.ckpt_path))
            else:
                if opt.ckpt_path.endswith(".pt"):
                    net.load_state_dict(torch.load(opt.ckpt_path))
                    print("[Info]: {:s} loaded.".format(os.path.abspath(opt.ckpt_path)))
        else:
            print("[Info]: network did not load a checkpoint!\n")

    ## ----- Set up the dataset and loader
    train_set = RegressorDataset(opt.feat_path, opt.score_path)
    if opt.debug:
        opt.nw = 0
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.nw,  # 4, 8
                                               pin_memory=True)

    ## ----- Set up optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)

    lf = lambda x: (((1 + math.cos(x * math.pi / opt.n_epoch)) / 2) ** 1.0) \
                   * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ## ----- Set up loss function
    loss_func = torch.nn.MSELoss().to(dev)

    ## ---------- Start training...
    for epoch in range(opt.n_epoch):
        epoch_loss = []

        for batch, (feat, score) in enumerate(train_loader):
            feat, score = feat.to(dev), score.to(dev)

            pred = net.forward(feat)

            loss = loss_func(pred, score) / opt.batch_size
            if (batch + 1) % opt.print_freq == 0:
                print("Epoch {:03d} | batch {:03d} | loss {:5.3f} | lr {:.5f}"
                      .format(epoch + 1,
                              batch + 1,
                              loss.item(),
                              optimizer.param_groups[0]["lr"]))

            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("[Info]: Average loss of epoch {:04d} is {:.5f}"
              .format(epoch + 1,
                      np.mean(np.array(epoch_loss))))

        ## ---------- update learning rate
        scheduler.step()

        if (epoch + 1) % opt.save_freq == 0:
            save_ckpt_path = opt.ckpt_dir + "/regressor_latest.pt"
            save_ckpt_path = os.path.abspath(save_ckpt_path)
            torch.save(net.state_dict(), save_ckpt_path)
            print("[Info]: {:s} saved.".format(save_ckpt_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--feat_path",
                        type=str,
                        default="./data/my_feats_koniq10k.npy",
                        help="")
    parser.add_argument("--score_path",
                        type=str,
                        default="./data/my_scores_koniq10k.npy",
                        help="")
    parser.add_argument("--name",
                        type=str,
                        default="koniq10k",
                        help="")
    parser.add_argument("--ckpt_dir",
                        type=str,
                        default="./models/")
    parser.add_argument("--reload",
                        type=bool,
                        default=True,
                        help="")

    ## ./models/regressor_latest.pt
    parser.add_argument("--ckpt_path",
                        type=str,
                        default="./models/regressor_latest.pt",
                        help="")
    parser.add_argument("--input_size",
                        type=int,
                        default=4096,
                        help="")
    parser.add_argument("--hidden_size",
                        type=int,
                        default=1024,
                        help="")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-4,
                        help="")
    parser.add_argument("--n_epoch",
                        type=int,
                        default=1000,
                        help="")
    parser.add_argument("--nw",
                        type=int,
                        default=4,
                        help="")
    parser.add_argument("--print_freq",
                        type=int,
                        default=10,
                        help="")
    parser.add_argument("--save_freq",
                        type=int,
                        default=30,
                        help="")
    parser.add_argument("--debug",
                        type=bool,
                        default=True,
                        help="")

    opt = parser.parse_args()

    run(opt)
    print("Done.")
