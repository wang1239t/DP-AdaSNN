import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode

import torch.nn.functional as F

import warnings

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors
import math

def autopad(k, p=None, d=1):

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)

@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)

class SpikeDFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self.lif = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class SpikeDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        def reg_hidden(c):
            return max((16, c // 4, self.reg_max * 4))

        def cls_hidden(c):
            return max(c, min(self.nc, 100))

        if self.nl == 4:

            reg_hidden_channels = [reg_hidden(ch[0])] + [reg_hidden(ch[1])] * (self.nl - 1)
            cls_hidden_channels = [cls_hidden(ch[0])] + [cls_hidden(ch[1])] * (self.nl - 1)
        else:

            reg_hidden_channels = [reg_hidden(ch[0])] * self.nl
            cls_hidden_channels = [cls_hidden(ch[0])] * self.nl
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(c_in, c_hidden, 3),
                SpikeConv(c_hidden, c_hidden, 3),
                SpikeConvWithoutBN(c_hidden, 4 * self.reg_max, 1),
            )
            for c_in, c_hidden in zip(ch, reg_hidden_channels)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                SpikeConv(c_in, c_hidden, 3),
                SpikeConv(c_hidden, c_hidden, 3),
                SpikeConvWithoutBN(c_hidden, self.nc, 1),
            )
            for c_in, c_hidden in zip(ch, cls_hidden_channels)
        )
        self.dfl = SpikeDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].mean(0).shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = x[i].mean(0)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):

            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self

        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].conv.bias.data[:] = 1.0
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class RepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            bias=False,
            group=1
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(

            nn.Conv2d(in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False),

            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SepRepConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            bias=False,
            group=1
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),

            nn.Conv2d(out_channel, out_channel, kernel_size, 1, 0, groups=out_channel, bias=False),
        )

        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SepAllConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,
                 padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)

        self.pwconv3 = SepRepConv(med_channels, dim)

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)

        self.lif1 = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )
        self.lif2 = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )
        self.lif3 = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,
                 padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)

        self.pwconv3 = SepRepConv(med_channels, dim)

        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)

        self.lif1 = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )
        self.lif2 = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )
        self.lif3 = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x

class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, full=False):
        super().__init__()

        self.full = full
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)
        self.mlp_ratio = mlp_ratio

        self.lif1 = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )
        self.lif2 = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )

        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio))

        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))

        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x

        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio) * C, H, W)

        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x

class MS_AllConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, group=False):
        super().__init__()

        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)

        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x

class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        self.lif = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, H // self.s, W // self.s)
        return x

class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()

        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MultiStepParametricLIFNode(
                detach_reset=True,
                v_reset=None,
                backend='torch'
            )

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_lif"):

            x = self.encode_lif(x)

        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x

class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        if x.shape[0] == 1:
            x = x.squeeze(0)

        if len(x.shape) == 4:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        elif len(x.shape) == 5:
            x = x
        return x

class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x

class SpikeConv(nn.Module):

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )
        self.bn = nn.BatchNorm2d(c2)
        self.s = s

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = H // self.s
        W_new = W // self.s
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeConvWithoutBN(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = MultiStepParametricLIFNode(
            detach_reset=True,
            v_reset=None,
            backend='torch'
        )

        self.s = s

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = H // self.s
        W_new = W // self.s
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeSPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))

class MS_Concat(nn.Module):

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        for i in range(len(x)):
            if x[i].dim() == 5:
                x[i] = x[i].mean(0)
        return torch.cat(x, self.d)
