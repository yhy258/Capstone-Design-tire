#%%
import os

from matplotlib import image

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torchvision import transforms
import torchvision.datasets as dsets
from torchvision.utils import save_image

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import fbeta_score

from tire_model import tire_model

from utils import *


def cv_main(args):
    
    # gpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loss_dict = {}
    val_loss_dict = {}
    val_accuracy_dict = {}
    
    model_dict = {}
    optimizer_dict = {}
    
    checkpoint_dir = './checkpoint/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir + f"fold_{args['fold_num']}", exist_ok=True)
    checkpoint_dir = checkpoint_dir + f"fold_{args['fold_num']}"
    start_epoch = 1
            
    transform_train = transforms.Compose([
        transforms.Resize([224,224], interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=(-20, 20), translate=(0.0, 0.3), scale=(0.8, 1.2)),
        transforms.ToTensor()
    ])

    # 데이터셋 로드
    train_data = dsets.ImageFolder(args['data_path'], transform=transform_train)
    
    # 학습 데이터와 검증 데이터 분할
    total_size = len(train_data)
    
    val_lens = [total_size // args['fold_num'] for i in range(args['fold_num'])]
    for i in range(total_size % args['fold_num']):
        val_lens[i] += 1
    
    print(total_size)
    print("fold val length: ",val_lens)
            
    indices = list(range(total_size))
    random.shuffle(indices)
    cumsum = make_cumsum(val_lens)
        
    # 이미지 확인
    index = 0
    img, label = train_data[index]
    img = img.numpy().transpose(1, 2, 0)

    print("정답: ", label)

    plt.imshow(img)
    plt.show()
    
    
    
    for k in range(args['fold_num']):
        torch.cuda.empty_cache() # GPU 캐시 데이터 삭제

        print(f"######### Now Fold : {k + 1}")

        # msg
        #         print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
        #               % (trll,trlr,trrl,trrr,vall,valr))
        
        train_indices, val_indices = make_fold_indices(k, args['fold_num'], indices, cumsum)
        
        train_set = torch.utils.data.dataset.Subset(train_data,train_indices)
        val_set = torch.utils.data.dataset.Subset(train_data,val_indices)
        
        print(len(train_set),len(val_set))
        print()
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'],
                                          shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args['batch_size'],
                                          shuffle=True, num_workers=4)

        train_loss_list = []
        val_loss_list = []
        val_accuracy_list = []
            
        model = tire_model(args["maskrcnn_path"], device)
        model.to(device)
        
        # optimizer = torch.optim.SGD(params=model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args["lr"], betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        criterion = nn.BCELoss()
        
    
        """Train part"""
        for epoch in range(start_epoch, args['epochs']+1):
            model.set_train()
            
            print("#################")
            print('[epoch %d]' % epoch)
            
            train_losses = []
            val_losses = []
            
            total = 0
            correct = 0
            
            for i, (img, label) in enumerate(train_loader):
                img = img.to(device)
                label = label.type(torch.FloatTensor)
                if args['label_smooth']:
                    label = torch.where(label > 0.5, label-0.1, label+0.1)
                label = label.to(device)
                
                out = model(img)

                loss = criterion(out, label.view(-1,1))
                train_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                            
            model.set_eval()
            
            with torch.no_grad():
                for i, (img, label) in enumerate(val_loader):
                    img = img.to(device)
                    label = label.type(torch.FloatTensor)
                    label = label.to(device)
                    label = label.view(-1,1)
                    
                    out = model(img)
                    
                    loss = criterion(out, label)
                    
                    val_losses.append(loss.item())
                    
                    predicted = out.data > args["threshold"]
                
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
        
            this_train_loss_mean = np.mean(train_losses)
            this_val_loss_mean = np.mean(val_losses)
            this_val_accuracy = (correct/total)*100
            train_loss_list.append(this_train_loss_mean)
            val_loss_list.append(this_val_loss_mean)
            val_accuracy_list.append(this_val_accuracy)
            
            scheduler.step(this_val_loss_mean)
            
            print(f"Train Loss : {this_train_loss_mean} \t Val Loss : {this_val_loss_mean} \t Val Accuracy : {this_val_accuracy}")
            
        train_loss_dict[f"{k + 1})_fold"] = train_loss_list
        val_loss_dict[f"{k + 1})_fold"] = val_loss_list
        val_accuracy_dict[f"{k + 1})_fold"] = val_accuracy_list
        model_dict[f"{k+1}_fold"] = model.state_dict()
        optimizer_dict[f"{k + 1})_fold"] = optimizer.state_dict()
        
    
        checkpoint = {
            'model': model_dict,
            'optimizer': optimizer_dict,
            'train_loss_dict': train_loss_dict,
            'val_loss_dict': val_loss_dict,
            'val_accuracy_dict' : val_accuracy_dict
        }
        checkpoint_name = os.path.join(checkpoint_dir, '{:d}_checkpoint.pth'.format(epoch))
        torch.save(checkpoint, checkpoint_name)
        print('checkpoint saved : ', checkpoint_name)
    # Loss Graph
    for i in range(args['fold_num']):
        
        plt.figure()
        plt.subplot(args['fold_num'],3,3*i+1)
        plt.title("train loss")
        x1 = np.arange(0, len(train_loss_dict[f"{i + 1})_fold"]))
        plt.plot(x1, train_loss_dict[f"{i + 1})_fold"])

        plt.subplot(args['fold_num'],3,3*i+2)
        plt.title("validation loss")
        x2 = np.arange(0, len(val_loss_dict[f"{i + 1})_fold"]))
        plt.plot(x2, val_loss_dict[f"{i + 1})_fold"])
        
        plt.subplot(args['fold_num'],3,3*i+3)
        plt.title("validation accuracy")
        x3 = np.arange(0, len(val_accuracy_dict[f"{i + 1})_fold"]))
        plt.plot(x3, val_accuracy_dict[f"{i + 1})_fold"])
        
    plt.tight_layout()
    plt.show()
    

        
    
def no_cv_main(args):
    
    # gpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_fbeta_list = []
    
    
    if args["checkpoint"] == None :
        checkpoint_dir = './checkpoint/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        start_epoch = 1
    else :
        print("start model load...")
        # 체크포인트 로드
        checkpoint = torch.load(args["checkpoint"], map_location=device)

        # 각종 파라미터 로드
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_loss_list = checkpoint['train_loss_list']
        val_loss_list = checkpoint['val_loss_list']
        val_accuracy_list = checkpoint['val_accuracy_list']
        start_epoch = checkpoint['epoch'] + 1

        print("model load end. start epoch : ", start_epoch)
        
    
    model = tire_model(args["maskrcnn_path"], device)
    model.to(device)
    
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args["lr"], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    criterion = nn.BCELoss()

    
    transform_train = transforms.Compose([
        transforms.Resize([224,224], interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=(-20, 20), translate=(0.0, 0.3), scale=(0.8, 1.2)),
        transforms.ToTensor()
    ])

    # 데이터셋 로드
    train_data = dsets.ImageFolder(args['data_path'], transform=transform_train)
    
    # 학습 데이터와 검증 데이터 분할
    n_val = int(len(train_data) * args['val_percent'])
    n_train = len(train_data) - n_val
    train_data, val_data = random_split(train_data, [n_train, n_val])
    print(f"Train Dataset Length : {n_train} \t Valid Dataset Length : {n_val}")
    
    # 이미지 확인
    index = 0
    img, label = train_data[index]
    img = img.numpy().transpose(1, 2, 0)

    print("정답: ", label)

    plt.imshow(img)
    plt.show()
    
    model.set_eval()
    
    # 데이터로더 생성
    train_loader = DataLoader(
                dataset=train_data,
                batch_size=args["batch_size"],
                shuffle=True)
    val_loader = DataLoader(
                     dataset=val_data,
                     batch_size=args["batch_size"],
                     shuffle=True)
    
    
    """Train part"""
    for epoch in range(start_epoch, args['epochs']+1):
        model.set_train()
        
        print("#################")
        print('[epoch %d]' % epoch)
        
        train_losses = []
        val_losses = []
        
        total = 0
        correct = 0
        
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.type(torch.FloatTensor)
            if args['label_smooth']:
                label = torch.where(label > 0.5, label-0.1, label+0.1)
            
            label = label.to(device)
            
            out = model(img)

            loss = criterion(out, label.view(-1,1))
            train_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
        model.set_eval()
        this_fbeta = []
        with torch.no_grad():
            for i, (img, label) in enumerate(val_loader):
                img = img.to(device)
                label = label.type(torch.FloatTensor)
                label = label.to(device)
                label = label.view(-1,1)
                
                out = model(img)
                
                loss = criterion(out, label)
                
                val_losses.append(loss.item())
                
                predicted = out.data > args["threshold"]
            
                total += label.size(0)
                correct += (predicted == label).sum().item()
                
                this_fbeta.append(fbeta_score(label.view(-1).tolist(), predicted.view(-1).tolist(), beta=args['beta'], average='micro'))
    
        this_train_loss_mean = np.mean(train_losses)
        this_val_loss_mean = np.mean(val_losses)
        this_val_accuracy = (correct/total)*100
        train_loss_list.append(this_train_loss_mean)
        val_loss_list.append(this_val_loss_mean)
        val_accuracy_list.append(this_val_accuracy)
        val_fbeta_list.append(np.mean(this_fbeta))
        
        scheduler.step(this_val_loss_mean)
        
        print(f"Train Loss : {this_train_loss_mean} \t Val Loss : {this_val_loss_mean} \t Val Accuracy : {this_val_accuracy} \t Val Fbeta : {np.mean(this_fbeta)}")
        
        if epoch % args["check_interval"] == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss_list': train_loss_list,
                'val_loss_list': val_loss_list,
                'val_accuracy_list' : val_accuracy_list,
                'val_fbeta_list' : val_fbeta_list
            }
            checkpoint_name = checkpoint_dir + '{:d}_checkpoint.pth'.format(epoch)
            torch.save(checkpoint, checkpoint_name)
            print('checkpoint saved : ', checkpoint_name)
    
    # Loss Graph

    plt.figure(figsize=(16, 10))
    plt.subplot(2,2,1)
    plt.title("train loss")
    x1 = np.arange(0, len(train_loss_list))
    plt.plot(x1, train_loss_list)

    plt.subplot(2,2,2)
    plt.title("validation loss")
    x2 = np.arange(0, len(val_loss_list))
    plt.plot(x2, val_loss_list)
    
    plt.subplot(2,2,3)
    plt.title("validation accuracy")
    x3 = np.arange(0, len(val_accuracy_list))
    plt.plot(x3, val_accuracy_list)
    
    plt.subplot(2,2,4)
    plt.title("validation Fbeta")
    x4 = np.arange(0, len(val_fbeta_list))
    plt.plot(x4, val_fbeta_list)
    plt.tight_layout()
    plt.savefig("result_.png", dpi=300)            

    plt.show()
    
        

if __name__=='__main__':
    print(torch.__version__)
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) multi gpu

    
    args_ = {
        "device_num" : "3",
        "data_path" : "/home/yhy258/Desktop/codes/타이어마모도데이터셋/Tiredata",
        "maskrcnn_path" : "/home/yhy258/Desktop/codes/타이어마모도데이터셋/maskrcnn_41.pth",
        "epochs" : 10,
        "batch_size" : 16,
        "lr" : 2e-5,
        'weight_decay' : 0.0001,
        "checkpoint" : None,
        "check_interval" : 100,
        "val_percent" : 0.2,
        "threshold" : 0.4,
        'fold_num' : 0,
        'label_smooth' : False,
        'beta' : 1.5
    }
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args_['device_num'])
    if args_["fold_num"] < 2:
        no_cv_main(args_)
    else :
        cv_main(args_)
    

# %%
