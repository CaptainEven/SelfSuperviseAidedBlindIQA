# encoding=utf-8

import os

import numpy as np
import scipy.io as sci_io


def gen_for_clive(root_path):
    """
    Generate dataset for CLIVE dataset
    """
    if not os.path.isdir(root_path):
        print("[Err]: invalid root path: {:s}".format(root_path))
        exit(-1)

    mat_dir = root_path + "/Data"
    if not os.path.isdir(mat_dir):
        print("[Err]: invalid mat dir path: {:s}.".format(mat_dir))
        exit(-1)

    AllImages_release_mat_path = mat_dir + "/AllImages_release.mat"
    if not os.path.isfile(AllImages_release_mat_path):
        print("[Err]: invalid path: {:s}.".format(AllImages_release_mat_path))
        exit(-1)
    AllMOS_release_mat_path = mat_dir + "/AllMOS_release.mat"
    if not os.path.isfile(AllMOS_release_mat_path):
        print("[Err]: invalid path: {:s}.".format(AllMOS_release_mat_path))
        exit(-1)

    mat_img = sci_io.loadmat(AllImages_release_mat_path)
    mat_mos = sci_io.loadmat(AllMOS_release_mat_path)
    # print(mat_img)
    # print(mat_mos)

    img_names = list(map(lambda x: x[0].tolist(),
                         np.squeeze(mat_img["AllImages_release"]).tolist()))
    print(img_names)
    img_scores = np.squeeze(mat_mos["AllMOS_release"])
    print(img_scores)


if __name__ == "__main__":
    gen_for_clive(root_path="/mnt/diske/ChallengeDB_release/ChallengeDB_release")
    print("Done.")
