# encoding=utf-8

import argparse
import os
import shutil

import numpy as np
from tqdm import tqdm
from gen_regressor_train_data_for_live import set_dev_and_net, get_feature


def generate(opt):
    """
    Generate dataset
    """
    root_path = opt.root_dir
    if not os.path.isdir(root_path):
        print("[Err]: invalid root path: {:s}".format(root_path))
        exit(-1)

    if opt.type == "small":
        img_dir = root_path + "/small/512x384"
        if not os.path.isdir(img_dir):
            print("[Err]: invalid image dir path: {:s}"
                  .format(os.path.abspath(img_dir)))
            exit(-1)
    elif opt.type == "full":
        img_dir = root_path + "/fill/1024x768"
        if not os.path.isdir(img_dir):
            print("[Err]: invalid image dir path: {:s}"
                  .format(os.path.abspath(img_dir)))
            exit(-1)
    else:
        print("[Info]: wrong data type!")
        exit(-1)

    name_score_f_path = root_path \
                        + "/koniq10k_scores_and_distributions.csv"
    if not os.path.isfile(name_score_f_path):
        print("[Info]: invalid file path: {:s}"
              .format(os.path.abspath(name_score_f_path)))
        exit(-1)

    if opt.viz:
        if os.path.isdir(opt.viz_dir):
            shutil.rmtree(opt.viz_dir)
        os.makedirs(opt.viz_dir)

    ## ----- Set up network and device
    net, dev = set_dev_and_net(opt)
    ## -----

    img_paths, scores, features = [], [], []

    with open(name_score_f_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        with tqdm(total=len(lines)) as progress_bar:
            for i, line in enumerate(lines):
                if i == 0:
                    field_names = line.split(",")
                    print(field_names)
                else:
                    line = line.strip()
                    fields = line.split(",")
                    score = float(fields[-1])
                    scores.append(score)

                    img_name = fields[0]
                    img_name = img_name.replace('"', '')
                    img_path = img_dir + "/" + img_name
                    if not os.path.isfile(img_path):
                        print("[Warning]: invalid file path: {:s}"
                              .format(img_path))
                        continue

                    if opt.logging:
                        print("\n[Info]: processing " + img_path,
                              "| Score: {:.3f}\n".format(score))

                    feature_vector = get_feature(net, img_path, dev)
                    features.append(np.squeeze(feature_vector))

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

                progress_bar.update()

    features = np.array(features)
    scores = np.array(scores)

    ## ----- serialize the training dataset
    if os.path.isdir(opt.out_dir):
        score_save_path = os.path.abspath(opt.out_dir + "/my_scores_{:s}.npy"
                                          .format(opt.name))
        feature_save_path = os.path.abspath(opt.out_dir + "/my_feats_{:s}.npy"
                                            .format(opt.name))

        np.save(score_save_path, scores)
        print("[Info]: {:s} written.".format(score_save_path))

        np.save(feature_save_path, features)
        print("[Info]: {:s} written.".format(feature_save_path))
    else:
        print("[Err]: invalid output dir path: {:s}".format(opt.out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir",
                        type=str,
                        default="/mnt/diske/koniq10k",
                        help="")
    parser.add_argument("--name",
                        type=str,
                        default="koniq10k",
                        help="")
    parser.add_argument("--out_dir",
                        type=str,
                        default="../data",
                        help="")
    parser.add_argument("--type",
                        type=str,
                        default="small",
                        help="")
    parser.add_argument("--ext",
                        type=str,
                        default=".bmp",
                        help="")
    parser.add_argument("--viz",
                        type=bool,
                        default=False,  # True
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

    generate(opt)
    print("Done.")
