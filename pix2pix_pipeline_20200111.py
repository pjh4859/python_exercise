import os
from glob import glob
import random
import torch
from torchvision import transforms as transforms
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dir_src, extension, is_train=True):
        super(CustomDataset, self).__init__()
        self.list_path_input = sorted(glob(os.path.join(dir_src, "Input", "*.{}".format(extension))))

        if is_train:
            self.list_path_input = sorted(glob(os.path.join(dir_src, "Target", "*.{}".format(extension))))
            self.idx = random.randint(10)

        transform = [transforms.ToTensor(),
                     # transforms.RandomCrop(),  # 이걸 쓰면 타겟과 인풋이미지의 자르는 부분이 달라서 비교를 할 수 없게 된다.
                     transforms.Lambda(lambda x: self.__random_crop(x)),
                     transforms.Normalize(mean=[0.5], std=[0.5])]

        self.transform = transforms.Compose(transform)
        self.is_train = is_train

    def __random_crop(self, x, size=32):
        return x[self.idx:self.idx + size, self.idx:self.idx + size]

    def __getitem__(self, index):
        input = Image.open(self.list_path_input[index])
        input = self.transform(input)

        if self.is_train:
            target = Image.open(self.list_path_target[index])
            target = self.transform(target)

            return input, target
        return input

    def __len__(self):
        return len(self.list_path_input)
