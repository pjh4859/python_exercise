import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm #타카둠 진행된 정도 볼 수 있음
import matplotlib.pyplot as plt
import os
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, n_ch,pre_activation=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias=False)
        self.bn1= nn.BatchNorm2d(n_ch) #배치노말라이제이션 채널마다 해줌 배치노말라이제이션은 가우시안 그림 변하는 것으로 설명했었음 너무 왜곡되지 않게
        self.act = nn.ReLU(inplace=True) #ReLU는 웨이트가 없음.
        #인플레이스 True 는 함수 통과한 애가 통과하기 전과 메모리 공간이 같은것 (메모리 아끼는데 쓰는듯)
        self.conv2=nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(n_ch)

        self.pre_activation = True if pre_activation else False
        # self. ~~ 해준 이유는 forward 에서 쓰기위해.

    def forward(self,x):
        if self.pre_activation:
            y = self.bn1(x)
            y = self.act(y)
            y = self.conv1(y)
            y = self.bn2(y)
            y = self.act(y)
            y = self.conv2(y)
            return x + y
        else:
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.act(y)
            y = self.conv2(y)
            y = self.bn2(y)
            return self.act(x + y)

class ResidualNetwork(nn.Module):
    def __init__(self, pre_activation=False):
        super(ResidualNetwork, self).__init__()
        # self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.act = nn.ReLU(True)
        #
        # self.rb1 = ResidualBlock(16)
        # self.rb2 = ResidualBlock(16)
        #
        # self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1,stride=2,biad=False)
        # self.bn2 = nn.BatchNorm2d(32)
        #
        # self.rb3 = ResidualBlock(32)
        # self.rb4 = ResidualBlock(32)
        #
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2 ,biad=False)
        # self.bn3 = nn.BatchNorm2d(64)
        #
        # self.rb5 = ResidualBlock(64)
        # self.rb6 = ResidualBlock(64)
        #
        # self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # self.linear = nn.Linear(64, 100)
        network = [nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)]
        if not pre_activation:
            network += [nn.BatchNorm2d(16),
                        nn.ReLU(True)]
        network += [ResidualBlock(16, pre_activation=pre_activation),
                    ResidualBlock(16, pre_activation=pre_activation),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1,stride=2, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    ResidualBlock(32, pre_activation=pre_activation),
                    ResidualBlock(32, pre_activation=pre_activation),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    ResidualBlock(64, pre_activation=pre_activation),
                    ResidualBlock(64, pre_activation=pre_activation)]
        if pre_activation:
            network += [nn.BatchNorm2d(64),
                        nn.ReLU(True)]
        network += [nn.AdaptiveAvgPool2d((1,1)),
                    View((64)),
                    nn.Linear(64,10)]
        self.network = nn.Sequential(*network) # 위의 network 처럼 해주면 class 안에 주석처리해준 것을 한 꼴이 됨.

        # network = []
        # network += [nn.Conv2d(1,16,kernel_size=3,padding=1,bias=False),  # 맨 처음 채널이 1인 것은 MNIST 가 흑백이미지라서
        #             # nn.BatchNorm2d(16),
        #             # nn.ReLU(True),
        #             ResidualBlock(16, pre_activation=True),
        #             ResidualBlock(16),
        #             nn.Conv2d(16, 32, kernel_size=3, padding=1,stride=2, bias=False),
        #             nn.BatchNorm2d(32),
        #             nn.ReLU(True),
        #             ResidualBlock(32),
        #             ResidualBlock(32),
        #             nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
        #             nn.BatchNorm2d(64),
        #             nn.ReLU(True),
        #             ResidualBlock(64),
        #             ResidualBlock(64),
        #             nn.BatchNorm2d(64), #추가
        #             nn.ReLU(True), #추가
        #             nn.AdaptiveAvgPool2d((1,1)), # 64x64x1x1
        #             View((64)),#64x64x1x1 >> 64x64 로 만들어줌. (2차원으로 넣어줘야 하기때문에)
        #             #Sequential 로 쓰려면 View 클래스를 정의해줘야함.
        #             nn.Linear(64,10)] #64x10 # 지금 MNIST 로 하기때문에 64,10

    def forward(self, x):
        return self.network(x)
        # x= self.conv1(x)
        # x= self.bn1(x)
        # x= self.act(x)
        #
        # x= self.rb1(x)
        # x= self.rb2(x)
        #
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.act(x)
        #
        # x = self.rb3(x)
        # x = self.rb4(x)
        #
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.act(x)
        #
        # x = self.rb5(x)
        # x = self.rb6(x)
        #
        # return x

class View(nn.Module):
    def __init__(self,*shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.shape[0],*self.shape) #0번째는 배치 디멘션 #64x64x1x1 >> 64x64