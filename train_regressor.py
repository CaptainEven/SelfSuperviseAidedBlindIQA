# encoding=utf-8

import argparse
import pickle

import numpy as np
from sklearn.linear_model import Ridge


def run(opt):
    """
    Run the regressor training
    """
    feat = np.load(opt.feat_path)
    scores = np.load(opt.ground_truth_path)

    # train regression
    reg = Ridge(alpha=opt.alpha).fit(feat, scores)
    pickle.dump(reg, open('lin_regressor.save', 'wb'))


def parse_args():
    """
    Argument Parser
    """
    parser = argparse.ArgumentParser(description="linear regressor")
    parser.add_argument('--feat_path',
                        type=str,
                        help='path to features file')
    parser.add_argument('--ground_truth_path',
                        type=str,
                        help='path to ground truth scores')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.1,
                        help='regularization coefficient')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = parse_args()
    run(args)
