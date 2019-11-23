import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, n_ch, growth_rate):
        super(DenseLayer, self).__init__()
        layer = []
        layer += [nn.BatchNorm2d(n_ch),
                  nn.ReLU(True),
                  nn.Conv2d(n_ch, growth_rate, 3, padding=1, bias=False)]
        # 여기에서 바이어스를 안걸어주는것은 위의 배치노말라이제이션에서 바이어스와 같은 역할을 해주기 때문에 안걸어 주는 것.
        self.layer = nn.Sequential(*layer)

    def forward(self, *inputs):
        x = self.layer(torch.cat(inputs, dim=1))  # (Denseblock 의 입력, 첫번째 레이어의 출력, 두 번째 레이어의 출력)
        return torch.cat((*inputs, x), dim=1)  # (16x 32x32)     (4x32x32)         (4x32x32)   >>  (24x32x32)


class DenseBlock(nn.Module):
    def __init__(self, n_layers, n_ch, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(n_layers):
            setattr(self, "Dense_layer_{}".format(i), DenseLayer(n_ch + i * growth_rate, growth_rate))
        # self.Dense_layer_0 = DenseLayer(n_ch + 0 * growth_rate, growth_rate)
        # self.Dense_layer_0 = DenseLayer(n_ch + 1 * growth_rate, growth_rate)
        # setattr 함수를 쓰면 위의 것을 한번에 씀.
        self.n_layers = n_layers

    def forward(self, x):
        for i in range(self.n_layers):
            x = getattr(self, "Dense_layer_{}".format(i))(x)
        return x


class Transition(nn.Module):
    def __init__(self, n_ch):
        super(Transition, self).__init__()
        layer = [nn.BatchNorm2d(n_ch),
                 nn.ReLU(True),
                 nn.Conv2d(n_ch, n_ch, kernel_size=1, bias=False),
                 nn.AvgPool2d(kernel_size=2, stride=2)]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, growth_rate, input_ch=1, n_classes=10):
        super(DenseNet, self).__init__()
        # assert depth in [40, 100]
        n_layers = 3
        init_ch = 8

        network = [nn.Conv2d(input_ch, init_ch, 3, padding=1, bias=False)]
        network += [DenseBlock(n_layers=n_layers, n_ch=init_ch, growth_rate=growth_rate)]  # 28x28
        n_ch = init_ch + growth_rate * n_layers

        network += [Transition(n_ch)]
        network += [DenseBlock(n_layers=n_layers, n_ch=n_ch, growth_rate=growth_rate)]  # 14x14
        n_ch = n_ch + growth_rate * n_layers

        network += [Transition(n_ch)]
        network += [DenseBlock(n_layers=n_layers, n_ch=n_ch, growth_rate=growth_rate)]  # 7x7
        n_ch = n_ch + growth_rate * n_layers

        network += [nn.BatchNorm2d(n_ch),
                    nn.ReLU(True),
                    nn.AdaptiveAvgPool2d(1),
                    View(-1),
                    nn.Linear(n_ch, n_classes)]

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
