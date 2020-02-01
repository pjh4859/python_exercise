import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_ch, patch_size):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        n_df = 64

        if patch_size == 1:
            model = [nn.Conv2d(n_ch, n_df, kernel_size=1, bias=False),
                     act]
            model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=1, bias=False),
                      nn.BatchNorm2d(2 * n_df),
                      act]
            model += [nn.Conv2d(2 * n_df, 1, kernel_size=1, bias=False)]

        elif patch_size == 16:
            model = [nn.Conv2d(n_ch, n_df, kernel_size=4, padding=1, stride=2, bias=False),
                     act]
            model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2, bias=False),
                      nn.BatchNorm2d(2 * n_df),
                      act]
            model += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, bias=False),
                      nn.BatchNorm2d(4 * n_df),
                      act]
            model += [nn.Conv2d(4 * n_df, 1, kernel_size=1, bias=False)]

        elif patch_size == 70:
            model = [nn.Conv2d(n_ch, n_df, kernel_size=4, padding=1, stride=2, bias=False),
                     act]
            model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2, bias=False),
                      nn.BatchNorm2d(2 * n_df),
                      act]
            model += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2, bias=False),
                      nn.BatchNorm2d(4 * n_df),
                      act]
            model += [nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, bias=False),
                      nn.BatchNorm2d(8 * n_df),
                      act]
            model += [nn.Conv2d(8 * n_df, 1, kernel_size=1, bias=False)]

        elif patch_size == 286:
            model = [nn.Conv2d(n_ch, n_df, kernel_size=4, padding=1, stride=2, bias=False),
                     act]
            model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2, bias=False),
                      nn.BatchNorm2d(2 * n_df),
                      act]
            model += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2, bias=False),
                      nn.BatchNorm2d(4 * n_df),
                      act]
            model += [nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=2, bias=False),
                      nn.BatchNorm2d(8 * n_df),
                      act]
            model += [nn.Conv2d(8 * n_df, 8 * n_df, kernel_size=4, padding=1, bias=False),
                      nn.BatchNorm2d(8 * n_df),
                      act]
            model += [nn.Conv2d(8 * n_df, 1, kernel_size=1, bias=False)]

        else:
            raise NotImplementedError("Invalid patch size{}. Please choose among [1, 16, 70, 186]".format(patch_size))

        model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, size, idx_max_ch=None):
        super(Generator, self).__init__()
        act_down = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        act_up = nn.ReLU(inplace=True)
        max_ch = 1024
        n_downsample = int(log2(size))
        n_gf = 64
        norm = nn.BatchNorm2d
        idx_max_ch = int(log2(max_ch // n_gf))

        for i in range(n_downsample):
            if i == 0:
                down_block = [nn.Conv2d(1, n_gf, kernel_size=4, padding=1, stride=2, bias=False)]
                up_block = [act_up,
                            nn.ConvTranspose2d(2 * n_gf, 3, kernel_size=4, padding=1, stride=2, bias=False),
                            nn.Tanh()]

            elif 1 <= i <= idx_max_ch:
                down_block = [act_down,
                              nn.Conv2d(n_gf, 2 * n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                              norm(2 * n_gf)]
                up_block = [act_up,
                            nn.ConvTranspose2d(4 * n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                            norm(n_gf)]


            elif idx_max_ch < i < n_downsample - 4:
                down_block = [act_down,
                              nn.Conv2d(n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                              norm(n_gf)]
                up_block = [act_up,
                            nn.ConvTranspose2d(2 * n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                            norm(n_gf)]

            elif n_downsample - 4 <= i < n_downsample - 1:
                down_block = [act_down,
                              nn.Conv2d(n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                              norm(n_gf)]
                up_block = [act_up,
                            nn.ConvTranspose2d(2 * n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                            norm(n_gf),
                            nn.Dropout2d(0.5, inplace=True)]

            else:
                down_block = [act_down,
                              nn.Conv2d(n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False)]
                up_block = [act_up,
                            nn.ConvTranspose2d(n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                            nn.Dropout2d(0.5, inplace=True)]

            self.add_module("Down_block_{}".format(i), nn.Sequential(*down_block))
            # add_module은 변수의 이름을 추가해주는 역할.
            # Down_block_0 = nn.Sequential(*down_block) 이렇게 하는 역할.
            self.add_module("Up_block_{}".format(i), nn.Sequential(*up_block))
            n_gf *= 2 if n_gf < max_ch and i != 0 else 1

        self.n_downsample = n_downsample

    def forward(self, x):
        layers = [x]
        for i in range(self.n_downsample):
            layers += [getattr(self, "Down_block_{}".format(i))(layers[-1])]
        x = getattr(self, "Up_block_{}".format(self.n_downsample - 1))(layers[-1])

        for i in range(self.n_downsample - 1, 0, -1):
            x = getattr(self, "Up_block_{}".format(i - 1))(torch.cat([x, layers[i]], dim=1))

        return x
# 스킵 커넥션은 원본의 형상을 유지하기 위해서이다.
# 앞쪽 단의 애를 뒷쪽 단의 애들과 연결해주면 gradient 가 path 를 따라 넘어가게 되는데,
