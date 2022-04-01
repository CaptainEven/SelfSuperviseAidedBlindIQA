# encoding=utf-8

import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import Ridge


def run(opt):
    """
    Run the regressor training
    """
    feats = np.load(opt.feat_path)
    print("[Info]: {:s} loaded.".format(os.path.abspath(opt.feat_path)))

    scores = np.load(opt.ground_truth_path)
    print("[Info]: {:s} loaded.".format(os.path.abspath(opt.ground_truth_path)))

    # train regression
    print("[Info]: start training score regressor...")
    reg = Ridge(alpha=opt.alpha).fit(feats, scores)
    print("[Info]: score regressor training done.")

    with open(opt.save_path, 'wb') as f:
        pickle.dump(reg, f)
    print("[Info]: {:s} saved.".format(os.path.abspath(opt.save_path)))


def parse_args():
    """
    Argument Parser
    """
    parser = argparse.ArgumentParser(description="linear regressor")

    parser.add_argument('--feat_path',
                        type=str,
                        default="./data/my_feats_koniq10k.npy",
                        help='path to features file')
    parser.add_argument('--ground_truth_path',
                        type=str,
                        default="./data/my_scores_koniq10k.npy",
                        help='path to ground truth scores')
    parser.add_argument("--save_path",
                        type=str,
                        default="./models/my_koniq10_small.save")
    parser.add_argument('--alpha',
                        type=float,
                        default=0.1,
                        help='regularization coefficient')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()
    run(opt)
