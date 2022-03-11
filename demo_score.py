# encoding=utf-8

import argparse
import os
import pickle
import shutil

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from modules.CONTRIQUE_model import CONTRIQUE_model
from modules.network import get_network

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_img(img_path):
    # load image
    image = Image.open(img_path)

    # downscale image by 2
    sz = image.size
    image_ds = image.resize((sz[0] // 2, sz[1] // 2))

    # transform to tensor
    image = transforms.ToTensor()(image).unsqueeze(0).cuda()
    image_ds = transforms.ToTensor()(image_ds).unsqueeze(0).cuda()
    return image, image_ds


def get_score(model, regressor, image, image_ds):
    """
    extract feature and regress score
    """
    # extract features
    model.eval()
    with torch.no_grad():
        _, _, _, _, model_feat, model_feat_2, _, _ = model.forward(image, image_ds)
    feat = np.hstack((model_feat.detach().cpu().numpy(), \
                      model_feat_2.detach().cpu().numpy()))

    # regress score
    score = regressor.predict(feat)[0]
    return score


def run(opt):
    """
    Run the demo
    """

    # load CONTRIQUE Model
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model(opt, encoder, 2048)
    model.load_state_dict(torch.load(opt.model_path, map_location=opt.device.type))
    model = model.to(opt.device)

    # load regressor model
    regressor = pickle.load(open(opt.linear_regressor_path, 'rb'))
    print("Models loaded.")

    print("Start scoring...")
    if os.path.isfile(opt.input_path) and opt.input_path.endswith(".txt"):
        if opt.viz:
            if not os.path.isdir(opt.viz_dir):
                try:
                    os.makedirs(opt.viz_dir)
                except Exception as e:
                    print(e)
                opt.viz_dir = os.path.abspath(opt.viz_dir)
                print("{:s} made.".format(opt.viz_dir))

        with open(opt.list_path) as f:
            lines = f.readlines()
            lines.sort()
            cnt = 0
            with tqdm(total=len(lines)) as progress_bar:
                for line in lines:
                    img_path = line.strip()
                    if not os.path.isfile(img_path):
                        print("[Warning]: invalid file path: {:s}".format(img_path))
                        continue

                    image, image_ds = load_img(img_path)
                    score = get_score(model, regressor, image, image_ds)
                    # print(score)

                    if opt.viz:
                        img_name = os.path.split(img_path)[-1]
                        pre, ext = img_name.split(".")
                        viz_save_path = opt.viz_dir + "/" \
                                        + "{:.3f}_{:s}_{:d}".format(score, pre, cnt) \
                                        + ".{:s}".format(ext)
                        if not os.path.isfile(viz_save_path):
                            shutil.copyfile(img_path, viz_save_path)

                    cnt += 1
                    progress_bar.update()

    elif os.path.isfile(opt.input_path) and opt.input_path.endswith(".jpg"):
        image, image_ds = load_img(opt.input_path)
        score = get_score(model, regressor, image, image_ds)
        print(score)

    elif os.path.isdir(opt.input_path):
        root = opt.input_path
        sub_dirs = [root + "/" + x for x in os.listdir(root)
                    if os.path.isdir(root + "/" + x)]
        sub_dirs.sort()

        with tqdm(total=len(sub_dirs)) as progress_bar:
            for dir_path in sub_dirs:
                cnt = 0

                img_paths = [dir_path + "/" + x for x in os.listdir(dir_path)]
                print("\nTotal {:d} samples to be evaluated in {:s}...\n"
                      .format(len(img_paths), dir_path))

                for img_path in img_paths:
                    image, image_ds = load_img(img_path)
                    score = get_score(model, regressor, image, image_ds)
                    # print(score)

                    if opt.viz:
                        img_name = os.path.split(img_path)[-1]
                        if img_name.count(".") > 1:  # 跳过已经处理过的
                            continue
                        pre, ext = img_name.split(".")
                        viz_save_path =dir_path + "/" \
                                        + "{:.3f}_{:s}_{:d}".format(score, pre, cnt) \
                                        + ".{:s}".format(ext)
                        if not os.path.isfile(viz_save_path):
                            # shutil.copyfile(img_path, viz_save_path)
                            os.rename(img_path, viz_save_path)

                    cnt += 1

                progress_bar.update()


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--img_path',
    #                     type=str,
    #                     default='sample_images/60.bmp',
    #                     help='Path to image',
    #                     metavar='')
    # parser.add_argument("--list_path",
    #                     type=str,
    #                     default="./data/test.txt",
    #                     help="")
    parser.add_argument("--input_path",
                        type=str,
                        default="/mnt/diskd/even/plates",
                        help="")
    parser.add_argument("--viz",
                        type=bool,
                        default=True,
                        help="")
    parser.add_argument("--viz_dir",
                        type=str,
                        default="./visualize",
                        help="")
    parser.add_argument('--model_path',
                        type=str,
                        default='models/CONTRIQUE_checkpoint25.tar',
                        help='Path to trained CONTRIQUE model',
                        metavar='')
    parser.add_argument('--linear_regressor_path',
                        type=str,
                        default='models/CLIVE.save',
                        help='Path to trained linear regressor',
                        metavar='')

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    opt = parse_args()
    run(opt)
