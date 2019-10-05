import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm #타카도미 진행된 정도 볼 수 있음
import matplotlib.pyplot as plt
import os
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.input_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)  # 1x28x28 -> 16x28x28
        # o = (i - k +2p) //2 + 1
        self.layer_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, stride=2)  # 32x14x14
        self.layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=2)  # 64x7x7
        self.layer_3 = nn.AdaptiveAvgPool2d((1, 1))  # 64x1x1
        self.layer_4 = nn.Linear(in_features=64, out_features=10)  # 10개채널

    def forward(self, x):
        x1 = F.relu(self.input_layer(x)) #16x28x28
        x2 = F.relu(self.layer_1(x1)) #32x14x14
        x3 = F.relu(self.layer_2(x2)) #64x7x7

        x4 = self.layer_3(x3) #Bx64x1x1
        x5 = x4.view(x4.shape[0], 64)  # x4.shape : Bx64x1x1 >> Bx64  >>squeeze 1x64x1x1 >> 64
        output = self.layer_4(x5) # B(배치사이즈)x10
        return output

if __name__== '__main__': # 이 파일 불러올 때 특정부분만 불러오기 위해서 만듦: 간접적으로 불러왔을 때 이 행 밑으로는 실행 안함
    transforms = Compose([ToTensor(),  # -> [0,1]
                          Normalize(mean=[0.5], std=[0.5])])  # -> [-1,1]  #MNIST 데이터를 가져올 떄 이런 처리과정을 거쳐 가져오자

    dataset = MNIST(root='./datasets', download=True, transform=ToTensor(), train=True)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    model = CNN()# 위의 공장에서 모델을 만들어 낸 것. 이게 없으면 모델이 없는것.
    criterion = nn.CrossEntropyLoss()  # loss function 크로스 엔트로피로스 안에 소프트맥스가 들어있어서 여기서는 안함

    optim = torch.optim.Adam(model.parameters(), lr=0.001) # 벌 받는 것. 최적화를 받을 대상이 꼭 들어가야 함 바로 웨이트들 모델 안의 파라미터들.
    # 위의 nn.Module 에서 정의된 것을 파라미터라는 변수가 웨이트를 나타냄
    # weight_new = weight_old - weight_gradient *lr

    if os.path.isfile('./CNN_model6.pt'):  # 인자가 있으면 true
        print("asdf")
        model_dict = torch.load('./CNN_model6.pt')['model_weight']
        model.load_state_dict(model_dict)
        adam_dict = torch.load('./CNN_model6.pt')['adam_weight']
        optim.load_state_dict(adam_dict)  # 모델이 있으면 불러와서 하는 것.

    list_loss = list()
    list_acc = []
    for epoch in range(10):
        for input, label in tqdm(data_loader):
            #label 32 답 한개씩 배치사이즈 만큼
            results = model(input) #32x10  파이토치가 이 10개 (0부터9) 중 가장 웨이트가 큰 것이 모델이 생각하는답이라 생각해 얘랑 라벨만 비교하는 듯(파이토치가 알아서 해준다)
            loss = criterion(results, label) #결과, 라벨 순서 꼭 지켜야함 안지키면 돌아가긴 함 근데 결과가 전혀 다름
            # list_loss.append(loss.item())

            optim.zero_grad() #잘못한 이전의 그라디언트를 0으로 만들어줌
            loss.backward() #웨이트들이게 그라디언트를 돌려줌
            optim.step()  # 여기서 파라미터들의 값이변함 (위 세 줄의 순서가 중요)
            list_loss.append(loss.detach().item()) #파이토치 실수형을 파이썬 실수형으로 바꿔준다.

            n_correct = torch.sum(torch.eq(torch.argmax(results, dim=1),label)).item()
            list_acc.append(n_correct / 32.0 * 100)
            print(np.mean(list_acc))
        break

    weight_dict1 = {'model_weight': model.state_dict(), 'adam_weight': optim.state_dict()}
    # torch.save(model,"./CNN_model.pt") #이렇게 저장하면 모델이 정의된 파일이랑 모델의 위치가 같아야 실행됨
    # troch.save(model.state_dict(),"./CNN_mocel.pt")  #이렇게 저장하면 모델이 저의된 파이썬 파일과 동일한 위치에 없어도 됨 (항상 이게 위에꺼보다 낫다)
    # torch.save(optim.state_dict(),"./adam.pt") #adam도 웨이트를 가지고있어서 저장할 경우. 위엣줄과 하면 파일이 두개 생성됨 따라서 하나로 만들기위해
    # 위에 weight_dict1 을 정의해줘서 하나의 파일로 저장
    torch.save(weight_dict1, "./CNN_model6.pt")

    plt.figure(1)
    plt.plot(list_acc)
    plt.xlabel("Iteration")
    plt.ylabel("accuracy")
    # plt.show()

    plt.figure(2)
    plt.plot(list_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    # plt.figure()
    # plt.plot(range(len(list_loss)), list_loss, linestyle='--')
    plt.show()

    # torch.save(model.state_dict(), "my_model3.pt")

    # 파라미터가 많으면 오버피팅이 일어날 수 있다. 필터가 많은것도 그런 현상을 낳을 수 있다.
    # 필터 = 채널 = 커널 같은 말
    # 인풋픽쳐가 1024x1024 같이 크면 메모리를 많이 먹고 파라미터 수를 늘릴 수가 없다 그래서 인풋픽쳐를 작게 해주기 위해
    # 스트라이드를 적용하기도 한다.
    # 데이터셋, 데이터로더, 모델, 로스펑션, 옵티마이져