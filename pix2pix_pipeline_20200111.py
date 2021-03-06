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
        self.is_train = is_train

    def __random_crop(self, x, size=256):
        return x[:, self.idx:self.idx + size, self.idx:self.idx + size]
        # 어디를 자를지 고르는 함수부분

    def __getitem__(self, index):
        input = Image.open(self.list_path_input[index])

        self.idx = random.randint(0, 10)

        transform = [transforms.ToTensor(),
                     # ToTensor pytorch가 인식할 수 있는 형태로 바꿔죽, 범위를 (0, 1) 로 바꿔주는 두가지 기능.
                     # transforms.RandomCrop(),  # 이걸 쓰면 타겟과 인풋이미지의 자르는 부분이 달라서 비교를 할 수 없게 된다.
                     transforms.Lambda(lambda x: self.__random_crop(x)),
                     # random_crop 은 랜덤하게 이미지를 자르는 것? 이걸 씀으로써 적은 수의 이미지를 뻥튀기할 수 있다.
                     transforms.Normalize(mean=[0.5], std=[0.5])]  # 여기서는 범위가 (-1, 1) 이 됨.
        transform = transforms.Compose(transform)

        if self.is_train:
            target = Image.open(self.list_path_target[index])

            return transform(input), transform(target)
        return transform(input)

    def __len__(self):
        return len(self.list_path_input)
    # 클래스의 인스턴스에 len 이라는 함수를쓸 수있는데, len 안에 정의된 데이터의 숫자를 알 수 있음.
