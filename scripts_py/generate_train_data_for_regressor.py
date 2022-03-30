# encoding=utf-8
import argparse
import os

import mat4py
import numpy as np
import scipy.io as sci_io
import torch

from modules.CONTRIQUE_model import DarknetModel
from modules.network import Darknet
from utils.utils import select_device, find_most_free_gpu, \
    load_img_pil_to_tensor

live_dmos_dict = {
    "jp2k": 227,  # 1: 227
    "jpeg": 233,  # 1: 233
    "wn": 174,  # 1: 1774
    "gblur": 174,  # 1: 174
    "fastfading": 174,  # 1: 174
}


def check_live_data(root_path, ext=".bmp"):
    """
    Check whether the number of images
    for each sub_dir is correct
    """
    def get_id(x):
        x = x.split(".")[0]
        x = int(x[3:])
        return x

    if not os.path.isdir(root_path):
        print("[Err]: invalid root path: {:s}".format(root_path))
        return

    img_path_list = []
    parsed_ref_names = []

    all_correct = 0
    for k, v in live_dmos_dict.items():
        print("[Info]: processing {:s}...".format(k))

        sub_dir_path = root_path + "/" + k
        if not os.path.isdir(sub_dir_path):
            print("[Err]: err sub_dir path:{:s}".format(sub_dir_path))
            continue

        ## ----- open info.txt
        sub_fields = []
        info_f_path = sub_dir_path + "/info.txt"
        with open(info_f_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                fields = line.split(" ")
                ref_name, img_name, score = fields
                # parsed_ref_names.append(ref_name)
                sub_fields.append((ref_name, img_name, score))
        sub_fields = sorted(sub_fields, key=lambda x: get_id(x[1]))
        # print(sub_fields)
        for fields in sub_fields:
            ref_name, img_name, score = fields
            parsed_ref_names.append(ref_name)

        cnt = 0
        img_names = [x for x in os.listdir(sub_dir_path) if x.endswith(ext)]
        img_names = sorted(img_names, key=lambda x: get_id(x))
        for img_name in img_names:
            # if not img_name.endswith(ext):
            #     continue

            img_path = sub_dir_path + "/" + img_name
            if not os.path.isfile(img_path):
                print("[Warning]: invalid image path: {:s}".format(img_path))
                continue
            else:
                cnt += 1
                img_path_list.append(img_path)

        if cnt == v:
            print("[Info]: Correct number of images for {:s}".format(k))
            all_correct += 1
        else:
            print("[Info]: In-correct number of images for {:s},"
                  " count: {:d}, value: {:d}".format(k, cnt, v))

    if all_correct == len(live_dmos_dict.items()):
        print("[Info]: all correct!")
        return img_path_list, parsed_ref_names


def count_live_imgs(root_path, ext=".bmp"):
    """
    Count distortion images and reference images
    """
    if not os.path.isdir(root_path):
        print("[Err]: invalid root path: {:s}".format(root_path))
        return

    cnt_dist, cnt_ref = 0, 0
    for sub_dir_name in os.listdir(root_path):
        sub_dir_path = root_path + "/" + sub_dir_name
        if os.path.isdir(sub_dir_path):
            if sub_dir_name == "refimgs":
                for img_name in os.listdir(sub_dir_path):
                    if img_name.endswith(ext):
                        cnt_ref += 1
                    else:
                        continue
            else:
                for img_name in os.listdir(sub_dir_path):
                    if img_name.endswith(ext):
                        cnt_dist += 1
                    else:
                        continue
        elif os.path.isfile(sub_dir_path):
            print("[Info]: skip {:s}".format(sub_dir_path))
            continue

    print("Total {:d} distortion images.".format(cnt_dist))
    print("Total {:d} reference images.".format(cnt_ref))


def load_live_mats(root_path, img_path_list, parsed_ref_names):
    """
    test read in mat file
    """
    if not os.path.isdir(root_path):
        print("[Err]: invalid mat dir path: {:s}".format(root_path))
        exit(-1)

    mat_dmos_path = root_path + "/dmos.mat"
    mat_ref_names_path = root_path + "/refnames_all.mat"
    if not os.path.isfile(mat_dmos_path):
        print("[Err]: invalid path: {:s}".format(mat_dmos_path))
        exit(-1)
    if not os.path.isfile(mat_ref_names_path):
        print("[Err]: invalid path: {:s}".format(mat_ref_names_path))
        exit(-1)

    mat_dmos = sci_io.loadmat(mat_dmos_path)
    for item in mat_dmos:
        print(item)

    # mat_ref_names = mat4py.loadmat(mat_ref_names_path)
    mat_ref_names = sci_io.loadmat(mat_ref_names_path)
    ref_names = mat_ref_names["refnames_all"]
    ref_names = ref_names[0]
    ref_names = list(map(lambda x: x.tolist()[0], ref_names))
    # print(ref_names)

    # print(img_path_list)
    # print(parsed_ref_names)

    correct_cnt, wrong_cnt = 0, 0
    for item1, item2 in zip(parsed_ref_names, ref_names):
        if item1 == item2:
            correct_cnt += 1
        else:
            wrong_cnt += 1
            print("[Info]: Not equal: {:s}, {:s}".format(item1, item2))
    if correct_cnt == len(ref_names):
        print("[Info]: parsing correctly!")
    else:
        print("[Info]: total {:d} correct.".format(correct_cnt))
        print("[Info]: total {:d} wrong.".format(wrong_cnt))


def build_net(opt):
    """
    Run the demo
    """
    ## ----- Set up device
    dev = str(find_most_free_gpu())
    print("[Info]: Using GPU {:s}.".format(dev))
    dev = select_device(dev)
    opt.device = dev
    dev = opt.device

    ## ----- Set up the net
    encoder = Darknet(cfg_path=opt.backbone_cfg, net_size=opt.image_size)
    net = DarknetModel(opt, encoder, opt.n_features)

    ## ----- Load Darknet backbone encoder
    print("Loading checkpoint {:s}...".format(opt.model_path))
    net.load_state_dict(torch.load(opt.model_path,
                                   map_location=opt.device.type))
    net = net.to(dev)
    net.eval()

    return net


def get_feature(net, img_path, dev):
    """
    extract feature vector
    """
    image, image_ds = load_img_pil_to_tensor(img_path, dev)
    with torch.no_grad():
        _, _, _, _, feat1, feat2, _, _ = net.forward(image, image_ds)
    feat = np.hstack((feat1.detach().cpu().numpy(),
                      feat2.detach().cpu().numpy()))
    return feat  # 1×4096


def gen_train_data_for_live(opt):
    """
    Generate training dataset for LIVE dataset
    """
    if not os.path.isdir(opt.root_dir):
        print("[Err]: invalid root dir path: {:s}".format(opt.root_dir))
        exit(-1)

    ## ----- Set up device
    dev = str(find_most_free_gpu())
    print("[Info]: Using GPU {:s}.".format(dev))
    dev = select_device(dev)
    opt.device = dev
    dev = opt.device

    ## ----- Build the network
    net = build_net(opt)

    ## ----- Traverse root dir and generate training dataset
    for sub_dir_name in os.listdir(opt.root_dir):
        sub_dir_path = opt.root_dir + "/" + sub_dir_name
        if not os.path.isdir(sub_dir_path):
            print("[Warning]: {:s} is not a dir.".format(sub_dir_path))
            continue

        info_f_path = sub_dir_path + "/info.txt"
        if not os.path.isfile(info_f_path):
            print("[Warning]: invalid info file path: {:s}".format(info_f_path))
            continue

        cnt = 0
        feature_list = []
        score_list = []
        img_path_list = []
        with open(info_f_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                fields = line.split(" ")
                try:
                    ref_name, img_name, score = fields
                except Exception as e:
                    print(e)

                img_path = sub_dir_path + "/" + img_name
                if not os.path.isfile(img_path):
                    print("[Warning]: invalid image path: {:s}".format(img_path))
                    continue

                feature_vector = get_feature(net, img_path, dev)
                feature_vector = np.squeeze(feature_vector)
                feature_list.append(feature_vector)

                score = float(score)
                score = 100.0 if score == 0.0 else score
                score_list.append(score)

                img_path_list.append(img_path)

                cnt += 1

        print("[Info]: total {:d} image samples found.".format(cnt))
        feature_np = np.array(feature_list)
        score_np = np.array(score_list)
        print(feature_np.shape)
        print(score_np.shape)

        ## ----- serialize the training dataset
        if os.path.isdir(opt.out_dir):
            feat_save_path = os.path.abspath(opt.out_dir + "/feats.npy")
            score_save_path = os.path.abspath(opt.out_dir + "/scores.npy")
            np.save(feat_save_path, feature_np)
            np.save(score_save_path, score_np)
            print("[Info]: {:s} written.".format(feat_save_path))
            print("[Info]: {:s} written.".format(score_save_path))
        else:
            print("[Err]: invalid output dir path: {:s}".format(opt.out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir",
                        type=str,
                        default="/mnt/diske/databaserelease2",
                        help="")
    parser.add_argument("--out_dir",
                        type=str,
                        default="../data",
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

    opt = parser.parse_args()

    ## ----- parse info.txt
    # gen_train_data_for_live(opt)

    # count_live_imgs(root_path="/mnt/diske/databaserelease2")
    img_path_list, parsed_ref_names = check_live_data(root_path="/mnt/diske/databaserelease2")

    ## ----- parse mat
    load_live_mats(root_path="/mnt/diske/databaserelease2",
                   img_path_list=img_path_list,
                   parsed_ref_names=parsed_ref_names)