import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from a20191005 import CNN
import numpy as np

# PIL Image >>> torch.tensor #자료형을 바꿔줘서 하는 것

dataset=MNIST(root='./datasets_test', download=True, train=False, transform=ToTensor())
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = CNN() #초기상태 모델이라 다르지만 레이어 같은 골격은 동일함
weight_dict = torch.load('./CNN_model1.pt')

for k,_ in weight_dict.items():
    print(k)

model_weight = weight_dict['model_weight']

for k, _ in model_weight.items():  # _를 변수로 둔 것은 관심이 없다는 뜻
    print(k)
# print(model_weight['input_layer.weight']) #input_layer 이름 같아야함 모델 불러오는 파일과
model.load_state_dict(model_weight)

list_acc = []
for input, label in tqdm(data_loader):
    output = model(input)

    n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item()
    print(n_correct_answers)
    list_acc.append(n_correct_answers / 32.0 * 100.)
    #print(np.mean(list_acc))
print("ACC:", np.mean(list_acc))