import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    该函数的作用是将输入通道数调整到离它最近的8的整数倍，这样做以后对硬件更加友好。
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    """
    模块名称：卷积BN激活函数模块
    """
    def __init__(self,
                 in_planes: int,   # 输入特征矩阵的channel
                 out_planes: int,  # 输出特征矩阵的channel
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,   # 用来控制当前的卷积是使用普通的卷积还是使用深度可分离卷积
                 norm_layer: Optional[Callable[..., nn.Module]] = None,   # 是efficientnet中的边结构
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # BN后面的激活函数
        padding = (kernel_size - 1) // 2     # 根据kernel size 计算 padding
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)   # SiLU激活函数和Swish激活函数是一样的，只是名称 不一样

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),   # 由于使用边结构，所以将bias设置为False
                                               norm_layer(out_planes),  # norm_layer指的是边结构，它传入的参数是上一层特征矩阵输出的channel
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    """
    模块名称：SE模块
    """
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel  # 第一个1*1卷积升维后对应的channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor    # 第一个全连接层结点个数
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)   # kerenl size 1*1
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # output_size=(1,1) 对每个channel进行全局平均池化
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x   # scale代表每个输出channel的重要程度


class InvertedResidualConfig:
    """
    对应每个MBConv模块的配置参数
    """
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,          # 3 or 5
                 input_c: int,         # 输入MBConv模块的特征矩阵的channel
                 out_c: int,           # MBConv模块输出的特征矩阵的channel
                 expanded_ratio: int,  # 1 or 6
                 stride: int,          # 1 or 2  dw卷积对应的步长
                 use_se: bool,         # True
                 drop_rate: float,     # 对应MBConv模块的Dropout层
                 index: str,           # 1a, 2a, 2b, ...  记录当前MBConv模块的名称
                 width_coefficient: float):  # 网络宽度方向上的倍率因子
        self.input_c = self.adjust_channels(input_c, width_coefficient)   # 真实的input_channel
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    """
    模块名称：MBConv模块
    """
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:  # 判断dw卷积的步长是否在1和2中
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)  # 判断是否是使用short cut连接

        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish

        # expand  1*1 卷积模块
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise dw卷积模块
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:  # 添加SE模块
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project  搭建最后1*1的卷积层
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})  # 该模块没有激活函数，Identity的意思是不做任何处理
 
        self.block = nn.Sequential(layers)   # 搭建MBConv模块的主分支
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = nn.Dropout2d(p=cnf.drop_rate, inplace=True)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,    # 代表channel维度上的倍率因子，比如在EfficientNetB0中Stage1的3*3卷积层所使用的卷积核个数是32，
                                              #   那么在EfficientNetB6中就是32*1.8=57.6，接着取整到离它最近的8的证书倍即56，其它stage同理。
                 depth_coefficient: float,      # 代表depth维度上的倍率因子（仅针对stage2到stage8), 比如在EfficientNetB0中stage7的L=4, 那么
                                                # 在EfficientNetB6中就是4*2.6=10.4，接着向上取整即11。
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,   # 对应stage9, FC层前面的随机dropout比例
                 drop_connect_rate: float = 0.2,   # 对应MBConv模块的随机缩放比例？是从0慢慢增长到0.2
                 block: Optional[Callable[..., nn.Module]] = None, # MBConv模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None  # 对应普通的BN结构
                 ):
        super(EfficientNet, self).__init__()

        # 针对B0 结构配置参数，只记录离stage2到stage8的参数。kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],   # repeats 代表重复MBConv模块多少次。
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual   # MBConv模块

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)  # 其实是和BatchNorm2d是一样的，只是下次使用的时候不需要再传这两个参数。

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0   # 用来统计搭建MNBconv模块的次数。
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))  # 获取当前网络所有MBConv模块的重复次数。
        inverted_residual_setting = []   # 搭建所有MBConv模块的配置信息
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:   # 对应当前Stage的第一个模块
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] *= b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks，这个循环搭建出所有的MBConv结构。
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})  # index：模块名称，block: MBConv这个类。 视频第30分钟。

        # build top   构建stage9
        last_conv_input_c = inverted_residual_setting[-1].out_c  # 对应MBConv模块最后一个模块的输出channel.
        last_conv_output_c = adjust_channels(1280)  # 进行宽度方向上的调整
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,     # 构建1*1的卷积层。
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)   # 实例化stage1到stage9的1*1卷积层。
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 对应stage9的池化层。1 对应输出特征矩阵的高和宽都是1.
         
        # 定义分类器。
        classifier = []
        if dropout_rate > 0: # 如果大于0， 则需要dropout层。
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))  # 添加最后的全连接层。为什么在最后全连接层前面还可以加一个dropout层？官方这样实现的。
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)
