# encoding=utf-8

import os

import torch.nn as nn
import torchvision
from torch.nn import init

from modules.darknet_modules import build_modules
from utils.utils import parse_darknet_cfg


class Darknet(nn.Module):
    def __init__(self,
                 cfg_path,
                 net_size=(320, 320),
                 use_momentum=True,
                 init_weights=True):
        """
        :param cfg_path:
        :param net_size:
        :param use_momentum:
        """
        super(Darknet, self).__init__()

        if not os.path.isfile(cfg_path):
            print("[Err]: invalid cfg file path.")
            exit(-1)

        self.module_defs = parse_darknet_cfg(cfg_path)
        self.module_list, self.routs = build_modules(self.module_defs,
                                                     net_size,
                                                     cfg_path,
                                                     3,
                                                     use_momentum)
        if init_weights:
            self.init_weights()
            print("[Info]: network initiated done.")

        print("[Info]: network modules built done.\n")

    def init_layer_weights(self, layer):
        """
        :param layer:
        :return:
        """
        if isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            init.normal(layer.weight, std=1e-3)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def init_weights(self):
        """
        :return:
        """
        self.module_list.apply(self.init_layer_weights)

    def forward_once(self, x):
        """
        :param x:
        :return:
        """
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []  # 3(or 2) yolo layers correspond to 3(or 2) reid feature map layers

        # ---------- traverse the network(by traversing the module_list)
        use_output_layers = ['WeightedFeatureFusion',  # Shortcut(add)
                             'FeatureConcat',  # Route(concatenate)
                             'FeatureConcat_l',
                             'RouteGroup',
                             'ScaleChannel',
                             'ScaleChannels',  # my own implementation
                             'SAM']

        # ----- traverse forward
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in use_output_layers:  # sum, concat
                x = module.forward(x, out)

            elif name == 'YOLOLayer':  # x: current layer, out: previous layers output
                yolo_out.append(module.forward(x, out))

            # We need to process a shortcut layer combined with a activation layer
            # followed by a activation layer
            elif name == 'Sequential':
                for j, layer in enumerate(module):  # for debugging...
                    layer_name = layer.__class__.__name__

                    if layer_name in use_output_layers:
                        x = layer.forward(x, out)
                    else:
                        x = layer.forward(x)

            # run module directly, i.e. mtype = 'upsample', 'maxpool', 'batchnorm2d' etc.
            else:
                x = module.forward(x)

            # ----------- record previous output layers
            out.append(x if self.routs[i] else [])
            # out.append(x)  # for debugging: output every layer
        # ----------

        return x, out

    def forward(self, x):
        """
        :param x:
        :return:
        """
        ## ----- out: final output, outs: outputs of each layer
        out, layer_outs = self.forward_once(x)
        # out = out.view(x.shape[0], -1)
        out = out.squeeze()
        return out


def get_network(name, pretrained=False):
    """
    Pre-trained networks
    """
    network = {
        "VGG16": torchvision.models.vgg16(pretrained=pretrained),
        "VGG16_bn": torchvision.models.vgg16_bn(pretrained=pretrained),
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in network.keys():
        raise KeyError(f"{name} is not a valid network architecture")
    return network[name]
