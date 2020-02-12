if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from pix2pix_pipeline_20200111 import CustomDataset
    from pix2pix_networks_20200111 import Discriminator, Generator
    from tqdm import tqdm

    EPOCHS = 30

    dataset = CustomDataset(dir_src="C:/Users/PARKJaehee/Desktop/datasets",
                            extension="jpg",
                            is_train=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            num_workers=0,
                            shuffle=True)
    D = Discriminator(n_ch=1 + 3, patch_size=1)  # 1+3인 이유는 흑백 이미지와 컬러 이미지를 구별해 들어갔다고 의미
    G = Generator(size=256)  # 이 사이즈는 컴퓨터의 사양이 더 좋으면 키워도됨. 이거 바꿀때 pipeline 의 random_crop 부분도 같이 바꿔줘야함

    loss_gan = torch.nn.BCELoss()  # Binary Cross-entropy Loss
    loss_l1 = torch.nn.L1Loss()
    loss_weight = 10  # loss_l1 이 loss_gan 보다 비중이 크다. 그래서 저 비중을 loss_l1에 곱해줌.

    D_optim = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))

    for epoch in range(EPOCHS):
        for input, target in tqdm(dataloader):
            output = G(input)
            score_fake = D(torch.cat((input, output.detach()), dim=1))
            # detach 는 제네레이터로 나온 output 에 대해 gradient 를 적용되지 않게 한다.
            score_real = D(torch.cat((input, target), dim=1))  # cat 이 들어가는 이유는 페어로 들어가서이다.

            D_loss_fake = loss_gan(score_fake, torch.zeros_like(score_fake))
            # torch.zeros(score_fake)는 형태는 score_fake 이고 모든 값은 0으로 하는것. (0은 가짜같다는 것을 의미)
            D_loss_real = loss_gan(score_real, torch.ones_like(score_real))
            # torch.ones_like(score_real)은 형태는 score_real 이고 모든 값을 1로 하는 것. (1은 실제같다는 것을 의미)

            D_loss = (D_loss_fake + D_loss_real) * 0.5

            D_optim.zero_grad()  # 잘못한 이전의 그라디언트를 0으로 만들어줌
            D_loss.backward()  # loss 만큼 grad 를 나눠줌
            D_optim.step()  # 여기에서 나눠준 grad 를 업데이트

            # 여기까진 Discriminator 의 업데이트.

            score_fake = D(torch.cat((input, output), dim=1))
            G_loss = loss_gan(score_fake, torch.ones_like(score_fake)) + loss_weight * loss_l1(output, target)
            # 제네레이터는 1에 가까운 값 (진짜) 을 만들어야하기 때문에 ones_like 를 넣어야함.

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            print("Epoch: {}, G_loss: {:.3f}, D_loss{:.3f}".format(epoch, G_loss.detach().item(), D_loss.detach().item()))

        save_image(output.detach(), "C:/Users/PARKJaehee/Desktop/checkpoints/{}_fake.{}".format(epoch, "jpg"), normalize=True)
        # output은 detach를 꼭 해주어야한다. 이유는 learnable 파라미터를 가지고 있는 때문.
        # nomalize 는 pipeline 에서 노말라이즈를 해주었는가 물어보는것. 했으니 True 이다.
        save_image(target, "C:/Users/PARKJaehee/Desktop/checkpoints/{}_real.{}".format(epoch, "jpg"), normalize=True)
        # target 은 detach를 할 필요가 없다. 이유는 learnable 파라미터를 가지고 있는 애가 아니기 때문.
