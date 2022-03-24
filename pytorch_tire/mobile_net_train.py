#%%
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split

import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.models as models

if __name__ == '__main__':
    print(torch.__version__)
    #------------------------------------------
    # 파라미터 설정
    # 빈 칸 채우기

    # 전체 데이터를 몇 번이나 볼 것인가?
    start_epoch = 1
    epoch_num = 1000

    # 학습 시에 한번에 몇 개의 데이터를 볼 것인가?
    batch_size = 64

    # 검증 데이터 비율
    val_percent = 0.2

    # 학습률
    lr = 0.00015

    # 체크포인트 저장 경로
    checkpoint_dir = './checkpoint/'

    # 학습 재개 시 resume = True, resume_checkpoint='재개할 체크포인트 경로'
    resume = False
    resume_checkpoint = ''
    #------------------------------------------

    # gpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 생성
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
    model = mobilenet_v3_small
    model.to(device)

    # 최적화 기법 및 손실 함수
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    #------------------------------------------
    # 이전 체크포인트로부터 모델 로드
    if resume:
        print("start model load...")
        # 체크포인트 로드
        checkpoint = torch.load(resume_checkpoint, map_location=device)

        # 각종 파라미터 로드
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_loss_list = checkpoint['train_loss_list']
        val_loss_list = checkpoint['val_loss_list']
        start_epoch = checkpoint['epoch'] + 1
        batch_size = checkpoint['batch_size']

        print("model load end. start epoch : ", start_epoch)
    #------------------------------------------

    #------------------------------------------
    # SheepFaceImages dataset

    # 이미지 변형
    # 빈 칸 채우기
    transform_train = transforms.Compose([
        transforms.Resize([224,224], interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

    # 데이터셋 로드
    train_data = dsets.ImageFolder('data/tire/', transform=transform_train)

    # 학습 데이터와 검증 데이터 분할
    n_val = int(len(train_data) * val_percent)
    n_train = len(train_data) - n_val
    train_data, val_data = random_split(train_data, [n_train, n_val])

    # 이미지 확인
    index = 0
    img, label = train_data[index]
    img = img.numpy().transpose(1, 2, 0)

    print("정답: ", label)

    plt.imshow(img)
    plt.show()

    # 데이터로더 생성
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_data,
                     batch_size=batch_size,
                     shuffle=True)
    val_loader = torch.utils.data.DataLoader(
                     dataset=val_data,
                     batch_size=batch_size,
                     shuffle=True)
    #------------------------------------------

    for epoch in range(start_epoch, epoch_num+1):
        # 에폭 한 번마다 전체 데이터를 보게 됨
        print('[epoch %d]' % epoch)

        print('start training...')

        train_loss = 0.0
        val_loss = 0.0

        # 학습
        model.train()
    
        for i, (x, target) in enumerate(train_loader):
            x = x.to(device)
            target = target.to(device)
            
            out = model(x)

            # loss 계산
            loss = criterion(out, target)
            train_loss = train_loss + loss.item()

            # 가중치 갱신
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[%d: loss] %f' % (i, loss.item()))

        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        print('train loss : %f' % (avg_train_loss))

        print('start validation...')

        # 검증
        model.eval()

        with torch.no_grad():
            for i, (x, target) in enumerate(val_loader):
                x = x.to(device)
                target = target.to(device)
                
                out = model(x)

                # loss 계산
                loss = criterion(out, target)
                val_loss = val_loss + loss.item()

                print('[%d: loss] %f' % (i, loss.item()))

            avg_val_loss = val_loss / len(val_loader)
            val_loss_list.append(avg_val_loss)
            print('validation loss : %f' % (avg_val_loss))

        # 체크포인트 저장
        checkpoint_name = checkpoint_dir + '{:d}_checkpoint.pth'.format(epoch)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss_list': train_loss_list,
            'val_loss_list': val_loss_list,
            'batch_size': batch_size
        }

        torch.save(checkpoint, checkpoint_name)
        print('checkpoint saved : ', checkpoint_name)

    # 학습 그래프 그리기
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("train loss")
    x1 = np.arange(0, len(train_loss_list))
    plt.plot(x1, train_loss_list)

    plt.subplot(1,2,2)
    plt.title("validation loss")
    x2 = np.arange(0, len(val_loss_list))
    plt.plot(x2, val_loss_list)
    plt.show()
# %%
