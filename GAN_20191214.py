# GAN
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = []
        model += [nn.Linear(in_features=100, out_features=128),
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.5)]  # Dropout 은 확율(버리는 것), 0.5는 다음 레이어에 넘겨줄 확율. 각 노드가 0.5배율만큼 버림.
        model += [nn.Linear(in_features=128, out_features=256),
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.5)]
        model += [nn.Linear(in_features=256, out_features=28 * 28),
                  nn.Sigmoid()]
        self.model = nn.Sequential(*model)
        print(self)  # 모델 프린트

    def forward(self, x):
        return self.model(x)


class Discrimination(nn.Module):
    def __init__(self):
        super(Discrimination, self).__init__()
        model = []
        model += [nn.Linear(28 * 28, 256),
                  nn.ReLU(inplace=True)]

        model += [nn.Linear(256, 128),
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.5)]
        model += [nn.Linear(128, 1),
                  nn.Sigmoid()]
        self.model = nn.Sequential(*model)
        print(self)

    def forward(self, x):
        return self.model(x)


dataset = MNIST(root='.',
                transform=ToTensor(),
                download=True,
                train=True)

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          shuffle=True,
                                          batch_size=1)

G = Generator()
D = Discrimination()

# print(G)
# print(D)

G_optim = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.9))
D_optim = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.9))

# without batch dimension >> 1x28x28
# with batch dimension >> Bx1x28x28

def D_criterion(real_score, fake_score):
    return -torch.mean(torch.log(real_score + 1e-8) + torch.log(1 - fake_score + 1e-8))


def G_criterion(fake_score):
    return -torch.mean(torch.log(fake_score + 1e-8))


total_step = 0
for epoch in range(10):
    for real, _ in tqdm(data_loader):
        total_step += 1
        z = torch.rand(real.shape[0], 100)

        fake = G(z)
        real = real.view(real.shape[0], -1)  # Bx784

        fake_score = D(fake.detach())
        real_score = D(real)

        D_loss = D_criterion(real_score, fake_score)
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        fake_score = D(fake)

        G_loss = G_criterion(fake_score)
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

        if total_step % 100 == 0:
            save_image(fake.view(fake.shape[0], 1, 28, 28),
                       "{}_fake.png".format(epoch + 1),
                       nrow=1,
                       normalize=True,
                       range=(0, 1))

            save_image(real.view(fake.shape[0], 1, 28, 28),
                       "{}_real.png".format(epoch + 1),
                       nrow=1,
                       normalize=True,
                       range=(0, 1))


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)  # 0번째는 배치 디멘션 #64x64x1x1 >> 64x64
