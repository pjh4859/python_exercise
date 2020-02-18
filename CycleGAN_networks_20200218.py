'''
라벨 데이터가 없는 데이터를 언 슈퍼바이스드 러닝이라고 함
라벨 데이터가 있다면 fix2fix 가 성능이 더 좋다.
제네레이터가 가지는 loss 는 두 개. 1. 보냈을 때 예를들면 반고흐 그림 같은지 보는 loss 2. 다시 돌아왔을 때, 달라진 것과의 차이
Batch Normalization은 배치가 2개 이상...
BN 대신 Instance Normalization 을 쓰자. IN 은 각 채널에 대해서 평균과 standard deviation을 구한다. 개별로 시킨다.
BN는 주로 classification 에서 쓰고, IN은 주로 이미지 변환에서 쓴다.
Discriminator 의 값이 4x4 라면 receptive field 에 대해 생각해야한다. 4x4의 (1,1) 부분은 receptive field 의
왼쪽 위에 부분에 대해 진짜 같은지 가짜같은지 표현 한다.
patch discriminator는 모든 레이어가 컨볼루션으로 되어있으면 patch discriminator이다.
layer Normalization 이란 것도 있다.
'''
import torch.nn as nn
import torch
from math import log2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        in_channels = 3  # RGB 컬러라서 채널이 3.
        n_df = 64  # 디스크리미네이터의 첫 아웃풋 채널 수
        norm = nn.InstanceNorm2d

        network = [nn.Conv2d(in_channels, n_df, kernel_size=4, stride=2, padding=1), act]
        network += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, stride=2, padding=1), norm(2 * n_df), act]
        network += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, stride=2, padding=1), norm(4 * n_df), act]
        network += [nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1), norm(8 * n_df), act]
        network += [nn.Conv2d(8 * n_df, 1, 4, padding=1)]
        self.network = nn.Sequential(*network)

    def forword(self, x):
        return self.network(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        act = nn.ReLU(inplace=True)
        in_channels = 3
        out_channels = 3

        n_gf = 64
        n_RB = 6  # residual network 수
        norm = nn.InstanceNorm2d

        # 128 x 128
        network = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, n_gf, kernel_size=7), norm(n_gf), act]
        # ReflectionPad 는 패딩을 할 때 가장 가장자리의 것을 고려해서 패딩한다. 가장가리가 123 이면 123 321 이런식으로 321이 패딩된다.
        # 기본 패딩은 그냥0 값을 추가한다.
        network += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, stride=2, padding=1), norm(2 * n_gf), act]

        # 64x64
        # 66x66 by padding 좌우상하
        # o = [i - k + 2*p] / s + 1  >> [63/2]+1 = 32 0.5만큼 사라짐---------------(a)
        network += [nn.Conv2d(2 * n_gf, 4 * n_gf, kernel_size=3, stride=2, padding=1), norm(4 * n_gf), act]

        for i in range(n_RB):
            network += [ResidualBlock(4 * n_gf)]

        network += [nn.ConvTranspose2d(4 * n_gf, 2 * n_gf, 3, stride=2, padding=1, output_padding=1), norm(2 * n_gf),
                    act]
        # (a) 에 의해 사라지는 만큼 output_padding 을 해줘야 한다. 따라서 여기선 1 해줌.
        network += [nn.ConvTranspose2d(4 * n_gf, 2 * n_gf, 3, stride=2, padding=1, output_padding=1), norm(2 * n_gf), act]
        network += [nn.ReflectionPad2d(3),nn.Conv2d(n_gf, out_channels,7),nn.Tanh()]
        self.network = nn.Sequential(*network)

    def forward(self,x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_ch):
        super(ResidualBlock, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.InstanceNorm2d

        block = [nn.ReflectionPad2d(1), nn.Conv2d(n_ch, n_ch, 3), norm(n_ch), act]
        block += [nn.ReflectionPad2d(1), nn.Conv2d(n_ch, n_ch, 3), norm(n_ch)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)

def weight_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        module.weight.detach().normal_(mean=0., std=0.02)
        # 웨이트를 가우시안 에서뽑아라?
