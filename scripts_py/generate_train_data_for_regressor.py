# encoding=utf-8
import argparse
import os

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


def gen_train_data_for_live(opt):
    """
    test read in mat file
    """
    root_path = opt.root_dir
    if not os.path.isdir(root_path):
        print("[Err]: invalid mat dir path: {:s}".format(root_path))
        exit(-1)

    img_path_list, parsed_ref_names = check_live_data(root_path)

    mat_dmos_path = root_path + "/dmos.mat"
    mat_ref_names_path = root_path + "/refnames_all.mat"
    if not os.path.isfile(mat_dmos_path):
        print("[Err]: invalid path: {:s}".format(mat_dmos_path))
        exit(-1)
    if not os.path.isfile(mat_ref_names_path):
        print("[Err]: invalid path: {:s}".format(mat_ref_names_path))
        exit(-1)

    ## ----- Set up device
    dev = str(find_most_free_gpu())
    print("[Info]: Using GPU {:s}.".format(dev))
    dev = select_device(dev)
    opt.device = dev
    dev = opt.device

    ## ----- Build the network
    net = build_net(opt)

    mat_dmos = sci_io.loadmat(mat_dmos_path)
    # for item in mat_dmos:
    # print(item)
    dmos_scores = np.squeeze(mat_dmos["dmos"])
    print(dmos_scores.shape)
    # dmos_scores = dmos_scores.tolist()

    # mat_ref_names = mat4py.loadmat(mat_ref_names_path)
    mat_ref_names = sci_io.loadmat(mat_ref_names_path)
    ref_names = mat_ref_names["refnames_all"]
    ref_names = ref_names[0]
    ref_names = list(map(lambda x: x.tolist()[0], ref_names))
    # print(ref_names)

    # print(img_path_list)
    # print(parsed_ref_names)

    correct_cnt, wrong_cnt = 0, 0
    for i, (item1, item2) in enumerate(zip(parsed_ref_names, ref_names)):
        if item1 == item2:
            correct_cnt += 1
        else:
            print("[Info]: Not equal: {:s}, {:s}".format(item1, item2))
            print("[Info]: wrong path: {:s}.".format(img_path_list[i]))
            wrong_cnt += 1

    if correct_cnt == len(ref_names):
        print("[Info]: All parsing correctly!")

        ## ----- generate LIVE training dataset
        if opt.use_ref:
            print("[Info]: total {:d} images samples.".format(len(ref_names)))
            ref_inds = np.where(dmos_scores == 0.0)
            dmos_scores[ref_inds] = 100.0
        else:  # do not use Reference image for training
            dist_inds = np.where(dmos_scores != 0.0)
            print("[Info]: total {:d} images samples.".format(len(dist_inds[0])))
            dmos_scores = dmos_scores[dist_inds]
            img_path_list = np.array(img_path_list)[dist_inds]

        feature_list = []
        for img_path, score in zip(img_path_list, dmos_scores):
            feature_vector = get_feature(net, img_path, dev)

            if opt.logging:
                print("[Info]: processing " + img_path,
                      "| Score: {:.3f}".format(float(score)))
                # print(feature_vector)

            feature_list.append(np.squeeze(feature_vector).tolist())
        features_np = np.array(feature_list)

        ## ----- serialize the training dataset
        if os.path.isdir(opt.out_dir):
            score_save_path = os.path.abspath(opt.out_dir + "/scores.npy")
            feature_save_path = os.path.abspath(opt.out_dir + "/feats.npy")

            np.save(score_save_path, dmos_scores)
            print("[Info]: {:s} written.".format(score_save_path))

            np.save(feature_save_path, features_np)
            print("[Info]: {:s} written.".format(feature_save_path))
        else:
            print("[Err]: invalid output dir path: {:s}".format(opt.out_dir))

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
    print("[Info]: Loading checkpoint {:s}...".format(opt.model_path))
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
    parser.add_argument("--use_ref",
                        type=bool,
                        default=True,
                        help="")
    parser.add_argument("--logging",
                        type=bool,
                        default=True,
                        help="")

    opt = parser.parse_args()

    ## ----- parse info.txt
    # gen_train_data_for_live(opt)

    # count_live_imgs(root_path="/mnt/diske/databaserelease2")
    # img_path_list, parsed_ref_names = check_live_data(root_path=opt.root_dir)

    ## ----- parse mat
    gen_train_data_for_live(opt)
