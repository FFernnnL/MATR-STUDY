import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class Conv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )

        # 判断是否需要进行额外的操作，当卷积核大小为1时，不进行额外操作
        if kernel_size == 1:
            self.ks_is_one = True
        else:
            self.ks_is_one = False
            self.oc = out_channels
            self.ks = kernel_size

            # 计算卷积核尺寸的平方除以2再加1的结果
            ws = kernel_size

            # 自适应平均池化层, 在进行池化操作时动态地调整输出的大小，而不是使用固定的池化窗口大小
            self.avg_pool = nn.AdaptiveAvgPool2d((ws, ws))

            # 用于调整卷积核的参数 Parameters used to adjust the convolution kernel
            self.pak = int((kernel_size * kernel_size) / 2 + 1)

            # 定义线性层，用于权重调整
            self.ce = nn.Linear(ws * ws, self.pak, False)
            self.ce_bn = nn.BatchNorm1d(in_channels)

            self.ci_bn2 = nn.BatchNorm1d(in_channels)
            self.relu = nn.ReLU(inplace=True)

            # 根据输入通道数确定划分的组数
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels

            # 定义线性层，用于调整输出通道数
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)

            # 定义线性层，用于卷积核尺寸调整
            self.gd = nn.Linear(self.pak, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.pak, kernel_size * kernel_size, False)

            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

            self.sig = nn.Sigmoid()

    def forward(self, x):
        if self.ks_is_one:
            # 如果卷积核大小为1，直接调用父类的卷积操作
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            """
            若卷积核大小不为1，则：
            1. 对输入进行自适应平均池化（nn.AdaptiveAvgPool2d）以将其调整为固定大小。
            2. 通过线性层（nn.Linear）、批归一化（nn.BatchNorm1d）和ReLU激活函数进行一系列操作，以调整卷积核的权重。
            3. 执行卷积操作，并将卷积核的权重与池化后的输入进行相乘，最终通过乘积的方式完成卷积操作。
            4. 对输入进行展开（nn.Unfold）以准备卷积计算。
            """

            b, c, h, w = x.size()
            weight = self.weight

            # 平均池化操作
            gl = self.avg_pool(x).view(b, c, -1)

            # 通过一系列操作得到卷积核权重的调整值
            out = self.ce(gl)
            ce2 = out
            out = self.ce_bn(out)
            out = self.relu(out)
            out = self.gd(out)

            # 根据输入通道数是否大于16确定组数，并进行相应的操作
            if self.g > 3:
                oc = self.ci(self.relu(self.ci_bn2(ce2).view(b, c // self.g, self.g, -1).transpose(2, 3))).transpose(2, 3).contiguous()
            else:
                oc = self.ci(self.relu(self.ci_bn2(ce2).transpose(2, 1))).transpose(2, 1).contiguous()

            # 重新调整输出形状
            oc = oc.view(b, self.oc, -1)
            oc = self.ci_bn(oc)
            oc = self.relu(oc)
            oc = self.gd2(oc)

            # Sigmoid操作，然后与输入的展开结果相乘，最后调整形状得到最终输出
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))
            x_un = self.unfold(x)
            b, _, l = x_un.size()
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)

            return torch.matmul(out, x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))
