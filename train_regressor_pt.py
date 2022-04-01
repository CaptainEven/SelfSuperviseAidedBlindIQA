# encoding=utf-8

import argparse
import os

import numpy as np
import torch
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
        feature = self.features[idx]
        score = self.scores[idx]

        return torch.tensor(feature), torch.tensor(score)

    def __len__(self):
        return self.n


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
    net = torch.nn.Linear(opt.input_size, 1)
    net = net.to(dev)

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

    ## ----- Set up loss function
    loss_func = torch.nn.MSELoss().to(dev)

    ## ---------- Start training...
    for epoch in range(opt.n_epoch):
        epoch_loss = []

        for batch, (feat, score) in enumerate(train_loader):
            feat, score = feat.to(dev), score.to(dev)

            pred = net.forward(feat)
            print(pred.shape)

            loss = loss_func(pred, score)
            if (batch + 1) % opt.print_freq == 0:
                print("Epoch {:03d} | batch {:03d} | loss {:5.3f}"
                      .format(epoch + 1, batch + 1, loss.item()))

            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("[Info]: Average loss of epoch {:04d} is {:.5f}"
              .format(epoch + 1,
                      np.mean(np.array(epoch_loss))))


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
    parser.add_argument("--input_size",
                        type=int,
                        default=4096,
                        help="")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="")
    parser.add_argument("--n_epoch",
                        type=int,
                        default=100,
                        help="")
    parser.add_argument("--nw",
                        type=int,
                        default=4,
                        help="")
    parser.add_argument("--print_freq",
                        type=int,
                        default=10,
                        help="")
    parser.add_argument("--debug",
                        type=bool,
                        default=True,
                        help="")

    opt = parser.parse_args()

    run(opt)
    print("Done.")
