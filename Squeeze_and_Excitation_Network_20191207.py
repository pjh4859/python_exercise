# Global average pooling 은 공간 사이즈를 1로 줄인다. 컴퓨테이션도 확 줄여준다.
#
# sigmoid 로 인해 0~1 사이의 값으로 변함.
# 블록을 지나 나온 attention map은 각각의 채널이 얼마나 가중치가 있는지 0~1사이 값으로 나타냄
# 거기에 input feature map을 곱해줘 output feature map이 된다.
import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, n_ch, reduction_ratio=16):
        super(SEModule, self).__init__()
        module = [nn.AdaptiveAvgPool2d(1),
                  nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(n_ch // reduction_ratio, n_ch, 1),
                  nn.Sigmoid()]
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return x * self.module(x)


class Plain(nn.Module):
    def __init__(self, SE=False):
        super(Plain, self).__init__()
        network = [nn.Conv2d(1, 16, 3, padding=1),
                   nn.ReLU(inplace=True)]
        network += [nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(inplace=True)]
        if SE:
            network += [SEModule(16, reduction_ratio=4)]

        network += [nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(True)]
        if SE:
            network += [SEModule(16, reduction_ratio=4)]

        network += [nn.Conv2d(16, 32, 3, padding=1, stride=2),
                    nn.ReLU(True)]
        network += [nn.Conv2d(32, 32, 3, padding=1),
                    nn.ReLU(True)]
        if SE:
            network += [SEModule(32, reduction_ratio=4)]
        network += [nn.Conv2d(32, 32, 3, padding=1),
                    nn.ReLU(True)]
        if SE:
            network += [SEModule(32, reduction_ratio=4)]

        network += [nn.Conv2d(32, 32, 3, padding=1),
                    nn.ReLU(True)]
        if SE:
            network += [SEModule(32, reduction_ratio=4)]

        network += [nn.AdaptiveAvgPool2d(1), View(-1), nn.Linear(32, 10)]
        self.network = nn.Sequential(*network)
        print(self)

    def forward(self, x):
        return self.network(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)  # 0번째는 배치 디멘션 #64x64x1x1 >> 64x64
