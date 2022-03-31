# encoding=utf-8

import argparse
import os
import shutil

import numpy as np
import scipy.io as sci_io

from gen_regressor_train_data_for_live import set_dev_and_net, get_feature


def gen_for_clive(opt):
    """
    Generate dataset for CLIVE dataset
    """
    root_path = opt.root_dir
    if not os.path.isdir(root_path):
        print("[Err]: invalid root path: {:s}".format(root_path))
        exit(-1)

    mat_dir = root_path + "/Data"
    if not os.path.isdir(mat_dir):
        print("[Err]: invalid mat dir path: {:s}.".format(mat_dir))
        exit(-1)

    img_dir = root_path + "/Images"
    if not os.path.isdir(img_dir):
        print("[Err]: invalid image dir path: {:s}.".format(img_dir))
        exit(-1)

    AllImages_release_mat_path = mat_dir + "/AllImages_release.mat"
    if not os.path.isfile(AllImages_release_mat_path):
        print("[Err]: invalid path: {:s}.".format(AllImages_release_mat_path))
        exit(-1)
    AllMOS_release_mat_path = mat_dir + "/AllMOS_release.mat"
    if not os.path.isfile(AllMOS_release_mat_path):
        print("[Err]: invalid path: {:s}.".format(AllMOS_release_mat_path))
        exit(-1)

    if opt.viz:
        if os.path.isdir(opt.viz_dir):
            shutil.rmtree(opt.viz_dir)
        os.makedirs(opt.viz_dir)

    ## ----- Set up network and device
    net, dev = set_dev_and_net(opt)
    ## -----

    mat_img = sci_io.loadmat(AllImages_release_mat_path)
    mat_mos = sci_io.loadmat(AllMOS_release_mat_path)
    # print(mat_img)
    # print(mat_mos)

    img_names = list(map(lambda x: x[0].tolist(),
                         np.squeeze(mat_img["AllImages_release"]).tolist()))
    print(img_names)

    ## ----- Build img paths
    img_paths = [img_dir + "/" + x for x in img_names
                 if os.path.isfile(img_dir + "/" + x)]
    print("[Info]: total {:d} images.".format(len(img_paths)))

    img_scores = np.squeeze(mat_mos["AllMOS_release"])
    # print(img_scores)

    feature_list = []
    for img_path, score in zip(img_paths, img_scores):
        feature_vector = get_feature(net, img_path, dev)

        if opt.logging:
            print("[Info]: processing " + img_path,
                  "| Score: {:.3f}".format(float(score)))
            # print(feature_vector)

        feature_list.append(np.squeeze(feature_vector).tolist())

        ## ----- visualize
        if opt.viz:
            img_name = os.path.split(img_path)[-1]
            viz_save_path = opt.viz_dir + "/" \
                            + img_name[:-len(opt.ext)] \
                            + "_{:.3f}".format(score) + opt.ext
            viz_save_path = os.path.abspath(viz_save_path)
            if not os.path.isfile(viz_save_path):
                shutil.copyfile(img_path, viz_save_path)
                print("{:s} saved.".format(viz_save_path))

    features_np = np.array(feature_list)

    ## ----- serialize the training dataset
    if os.path.isdir(opt.out_dir):
        score_save_path = os.path.abspath(opt.out_dir + "/scores_clive.npy")
        feature_save_path = os.path.abspath(opt.out_dir + "/feats_clive.npy")

        np.save(score_save_path, img_scores)
        print("[Info]: {:s} written.".format(score_save_path))

        np.save(feature_save_path, features_np)
        print("[Info]: {:s} written.".format(feature_save_path))
    else:
        print("[Err]: invalid output dir path: {:s}".format(opt.out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir",
                        type=str,
                        default="/mnt/diske/ChallengeDB_release/ChallengeDB_release",
                        help="")
    parser.add_argument("--out_dir",
                        type=str,
                        default="../data",
                        help="")
    parser.add_argument("--ext",
                        type=str,
                        default=".bmp",
                        help="")
    parser.add_argument("--viz",
                        type=bool,
                        default=True,
                        help="")
    parser.add_argument("--viz_dir",
                        type=str,
                        default="../visualize",
                        help="")
    parser.add_argument("--backbone_cfg",
                        type=str,
                        default="../yolov4_tiny_backbone.cfg",
                        help="")
    parser.add_argument("--n_features",
                        type=int,
                        default=2048,
                        help="")
    parser.add_argument('--image_size',
                        type=tuple,
                        default=(192, 64),  # (256, 256)
                        help='image size')
    ## checkpoint20.tar
    parser.add_argument('--model_path',
                        type=str,
                        default='../checkpoints/checkpoint0.tar',  # pretrained_res50.tar
                        help='Path to trained CONTRIQUE model',
                        metavar='')
    parser.add_argument("--use_ref",
                        type=bool,
                        default=False,
                        help="")
    parser.add_argument("--logging",
                        type=bool,
                        default=True,
                        help="")

    opt = parser.parse_args()

    gen_for_clive(opt)
    print("Done.")
