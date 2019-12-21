import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.z = nn.Linear(100, 200)
        self.y = nn.Linear(10, 1000)
        self.Linear = nn.Linear(1200, 28 * 28)
       # self.dropout = nn.Dropout(0.5)

    def forward(self, z, y):
        x = torch.cat((self.z(z), self.y(y)), dim=1)
        x = F.dropout(x, p=0.5)
        return nn.Sigmoid()(self.Linear(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.x = nn.Linear(28 * 28, 240)
        self.y = nn.Linear(10, 50)
        self.linear_1 = nn.Linear(290, 240)
        self.linear_2 = nn.Linear(240, 1)

    def forward(self, x, y):
        x = torch.cat((F.relu(self.x(x)), F.relu(self.y(y))), dim=1)
        x = F.dropout(self.linear_1(x), p=0.5)
        return nn.Sigmoid()(self.linear_2(x))


dataset = MNIST(root='.',
                transform=ToTensor(),
                download=True,
                train=True)

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          shuffle=True,
                                          batch_size=1)

G = Generator()
D = Discriminator()

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
    for real, label in tqdm(data_loader):
        total_step += 1

        real = real.view(1, -1)
        one_hot = torch.zeros(1, 10)  # 1x10 -> 모든값이 0
        one_hot.scatter_(dim=1, index=label.view(1, 1), src=torch.ones(1, 1))
        # 만약 label=0 -> [1,0,0,...,0]
        # 만약 label=9 -> [0,0,0,...,1]

        z = torch.rand(1, 100)
        fake = G(z, one_hot)

        # z = torch.rand(real.shape[0], 100)

        real = real.view(real.shape[0], -1)  # Bx784

        fake_score = D(fake.detach(), one_hot)
        real_score = D(real,one_hot)

        D_loss = D_criterion(real_score, fake_score)
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        fake_score = D(fake, one_hot)

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
