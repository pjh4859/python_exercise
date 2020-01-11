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

