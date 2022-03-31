# encoding=utf-8

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modules.CONTRIQUE_model import CONTRIQUE_model
from modules.network import get_network

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run(opt):
    """
    run the feature extraction
    """
    # load image
    img_path = os.path.abspath(opt.im_path)
    if os.path.isfile(img_path):
        image = Image.open(img_path)
    else:
        print("[Err]: invalid image path: {:s}".format(img_path))
        exit(-1)

    # downscale image by 2
    sz = image.size
    image_2 = image.resize((sz[0] // 2, sz[1] // 2))

    # transform to tensor
    image = transforms.ToTensor()(image).unsqueeze(0).cuda()
    image_2 = transforms.ToTensor()(image_2).unsqueeze(0).cuda()

    # load CONTRIQUE Model
    encoder = get_network('resnet50', pretrained=False)
    net = CONTRIQUE_model(opt, encoder, 2048)

    ckpt_path = os.path.abspath(opt.model_path)
    if os.path.isfile(ckpt_path):
        net.load_state_dict(torch.load(ckpt_path,
                                       map_location=opt.device.type))
    else:
        print("[Err]: invalid ckpt file path: {:s}".format(ckpt_path))
        exit(-1)
    net = net.to(opt.device)

    # extract features
    net.eval()
    with torch.no_grad():
        _, _, _, _, model_feat, model_feat_2, _, _ = net.forward(image, image_2)
    feat = np.hstack((model_feat.detach().cpu().numpy(),
                      model_feat_2.detach().cpu().numpy()))

    # save features model
    np.save(opt.feature_save_path, feat)
    print('Done')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--im_path',
                        type=str,
                        default='sample_images/img66.bmp',  # 33.bmp
                        help='Path to image',
                        metavar='')
    parser.add_argument('--model_path',
                        type=str,
                        default='models/checkpoint25.tar',
                        help='Path to trained CONTRIQUE model',
                        metavar='')
    parser.add_argument('--feature_save_path',
                        type=str,
                        default='features.npy',
                        help='Path to save_features', metavar='')
    opt = parser.parse_args()

    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return opt


if __name__ == '__main__':
    args = parse_args()
    run(args)
