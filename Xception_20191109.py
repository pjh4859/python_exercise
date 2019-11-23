'''
컨볼루션을 공간과 채널에 대해 나눌 수 있다. 이렇게 나누었을 때 보통 더 좋은 성능을 보이더라.
'''

import torch.nn as nn


class SeparateConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(SeparateConv, self).__init__()
        self.spatial_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                                      stride=stride, groups=in_ch)
        self.channel_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.bn_1 = nn.BatchNorm2d(in_ch)
        self.bn_2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.bn_1(x)
        x = self.channel_conv(x)
        x = self.bn_2(x)
        x = self.act(x)

        return x


class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        model = []
        model += [nn.Conv2d(1, 16, 3),
                  nn.BatchNorm2d(16),
                  nn.ReLU(True)]

        model += [SeparateConv(16, 16)]
        model += [SeparateConv(16, 32, stride=2)]
        model += [SeparateConv(32, 64, stride=2)]
        model += [nn.AdaptiveAvgPool2d((1, 1)),
                  View(-1),
                  nn.Linear(64, 10)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)  # 0번째는 배치 디멘션 #64x64x1x1 >> 64x64
