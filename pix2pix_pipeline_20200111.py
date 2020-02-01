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
        # glob 는 같은확장자의 애들을 한번에 긁어오는 데 씀, 리스트로 만들어준다.

        if is_train:
            self.list_path_target = sorted(glob(os.path.join(dir_src, "Target", "*.{}".format(extension))))
            self.idx = random.randint(10)

        transform = [transforms.ToTensor(),
                     # ToTensor pytorch가 인식할 수 있는 형태로 바꿔죽, 범위를 (0, 1) 로 바꿔주는 두가지 기능.
                     # transforms.RandomCrop(),  # 이걸 쓰면 타겟과 인풋이미지의 자르는 부분이 달라서 비교를 할 수 없게 된다.
                     transforms.Lambda(lambda x: self.__random_crop(x)),
                     # random_crop 은 랜덤하게 이미지를 자르는 것? 이걸 씀으로써 적은 수의 이미지를 뻥튀기할 수 있다.

                     transforms.Normalize(mean=[0.5], std=[0.5])]

        self.transform = transforms.Compose(transform)
        self.is_train = is_train

    def __random_crop(self, x, size=32):
        return x[self.idx:self.idx + size, self.idx:self.idx + size]
        # 어디를 자를지 고르는 함수부분

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
