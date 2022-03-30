# encoding=utf-8
import argparse
import os

import mat4py
import torch

from modules.CONTRIQUE_model import DarknetModel
from modules.network import Darknet
from utils.utils import select_device, find_most_free_gpu


def test_mat(mat_path):
    """
    test read in mat file
    """
    if not os.path.isfile(mat_path):
        print("[Err]: invalid mat file path: {:s}".format(mat_path))
        exit(-1)

    mat = mat4py.loadmat(mat_path)
    print(mat)


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


def gen_train_data_for_live(opt):
    """
    Generate training dataset for LIVE dataset
    """
    if not os.path.isdir(opt.root_dir):
        print("[Err]: invalid root dir path: {:s}".format(opt.root_dir))
        exit(-1)

    ## ----- Build the network
    net = build_net(opt)

    ## -----


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir",
                        type=str,
                        default="",
                        help="")
    parser.add_argument("--backbone_cfg",
                        type=str,
                        default="./yolov4_tiny_backbone.cfg",
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
                        default='checkpoints/checkpoint0.tar',  # pretrained_res50.tar
                        help='Path to trained CONTRIQUE model',
                        metavar='')

    opt = parser.parse_args()

    gen_train_data_for_live(opt)
