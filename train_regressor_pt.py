# encoding=utf-8

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class RegressorDataset(Dataset):
    def __init__(self, feat_npy_path, score_npy_path):
        feat_npy_path = os.path.abspath(feat_npy_path)
        score_npy_path = os.path.abspath(score_npy_path)

        if not os.path.isfile(feat_npy_path):
            print("[Info]: invalid feature file path {:s}"
                  .format(feat_npy_path))
            exit(-1)
        if not os.path.isfile(score_npy_path):
            print("[Info]: invalid score file path {:s}"
                  .format(score_npy_path))
            exit(-1)

        self.features = np.load(feat_npy_path)
        self.scores = np.load(score_npy_path)

    def __getitem__(self, idx):
        feature = self.features[idx]
        score = self.scores[idx]

    def __len__(self):
        return self.features.shape[0]


def run(opt):
    """
    Run the regressor training
    """
    pass


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

    opt = parser.parse_args()

    run(opt)
    print("Done.")
