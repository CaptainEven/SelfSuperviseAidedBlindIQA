# encoding=utf-8

import torch
import torch.nn as nn


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_idx, layers, stride):
        """
        :param anchors:
        :param nc:
        :param img_size:
        :param yolo_idx:
        :param layers:
        :param stride:
        """
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.index = yolo_idx  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y grid points
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def create_grids(self, ng=(13, 13), device='cpu'):
        """
        :param ng:
        :param device:
        :return:
        """
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, pred, out):
        """
        :param pred:
        :param out:
        :return:
        """
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.inde
            x, self.nl  # index in layers, number of layers
            pred = out[self.layers[i]]
            bs, _, ny, nx = pred.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), pred.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(pred[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            pred = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    pred += w[:, j:j + 1] * \
                            F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear',
                                          align_corners=False)

        else:
            bs, _, ny, nx = pred.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids(ng=(nx, ny), device=pred.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, na, ny, nx, no(classes + xywh))
        pred = pred.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return pred

        else:  # inference
            io = pred.clone()  # inference output

            # ---------- process pred to io
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh YOLO method
            io[..., :4] *= self.stride  # map from YOLO layer's scale to net input's scale
            torch.sigmoid_(io[..., 4:])  # sigmoid for confidence score and cls pred

            # gathered pred output: io: view [1, 3, 13, 13, 85] as [1, 507, 85]
            io = io.view(bs, -1, self.no)

            # return io, pred
            return io, pred

## route_lhalf
class FeatureConcat_l(nn.Module):
    def __init__(self, layers):
        """
        :param layers:
        """
        super(FeatureConcat_l, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        return torch.cat([outputs[i][:, :outputs[i].shape[1] // 2, :, :] for i in self.layers], 1) if self.multiple else \
            outputs[self.layers[0]][:, :outputs[self.layers[0]].shape[1] // 2, :, :]


## shortcut
class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        """
        :param layers:
        :param weight:
        """
        super(WeightedFeatureFusion, self).__init__()

        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


## route layer
class FeatureConcat(nn.Module):
    def __init__(self, layers):
        """
        :param layers:
        """
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class GAP(nn.Module):
    def __init__(self, dimension=1):
        """
        :param dimension:
        """
        super(GAP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return self.avg_pool(x)


class ScaleChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        """
        :param layers:
        """
        super(ScaleChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        a = outputs[self.layers[0]]
        return x.expand_as(a) * a
        # return torch.mul(a, x)


# SAM layer: ScaleSpatial
class SAM(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(SAM, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):  # using x as point-wise spacial attention[0, 1]
        a = outputs[self.layers[0]]  # using a as input feature
        return x * a  # point-wise multiplication


class MixDeConv2d(nn.Module):  # MixDeConv: Mixed Depthwise DeConvolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        """
        :param in_ch:
        :param out_ch:
        :param k:
        :param stride:
        :param dilation:
        :param bias:
        :param method:
        """
        super(MixDeConv2d, self).__init__()

        groups = len(k)
        if method == 'equal_ch':  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.ConvTranspose2d(in_channels=in_ch,
                                                   out_channels=ch[g],
                                                   kernel_size=k[g],
                                                   stride=stride,
                                                   padding=k[g] // 2,  # 'same' pad
                                                   dilation=dilation,
                                                   bias=bias) for g in range(groups)])

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return torch.cat([m(x) for m in self.m], 1)


# Dropout layer
class Dropout(nn.Module):
    def __init__(self, prob):
        """
        :param prob:
        """
        super(Dropout, self).__init__()
        self.prob = float(prob)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return F.dropout(x, p=self.prob)


class RouteGroup(nn.Module):
    def __init__(self, layers, groups, group_id):
        """
        :param layers:
        :param groups:
        :param group_id:
        """
        super(RouteGroup, self).__init__()
        self.layers = layers
        self.multi = len(layers) > 1
        self.groups = groups
        self.group_id = group_id

    def forward(self, x, outputs):
        """
        :param x:
        :param outputs:
        :return:
        """
        if self.multi:
            outs = []
            for layer in self.layers:
                out = torch.chunk(outputs[layer], self.groups, dim=1)
                outs.append(out[self.group_id])
            return torch.cat(outs, dim=1)
        else:
            out = torch.chunk(outputs[self.layers[0]], self.groups, dim=1)
            return out[self.group_id]