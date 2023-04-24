import os
import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms


class MyData(torch.utils.data.Dataset):
    def __init__(self, dir_A, dir_B, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if self.train:
            data_file_A = dir_A + "_train.txt"
            data_file_B = dir_B + "_train.txt"
        else:
            data_file_A = dir_A + "_test.txt"
            data_file_B = dir_B + "_test.txt"
        self.data=[]
        self.targets = []
        cnt = 0
        A, B = [], []
        with open(data_file_A) as f:
            for line in f:
                name = line.replace("\n", "")
                A.append(name)
        with open(data_file_B) as f:
            for line in f:
                name = line.replace("\n", "")
                B.append(name)
        random.shuffle(A)
        random.shuffle(B)
        for name_a, name_b in zip(A, B):
            path_a = os.path.join(dir_A, name_a)
            path_b = os.path.join(dir_B, name_b)
            img_a = cv2.imread(path_a)
            img_b = cv2.imread(path_b)
            self.data.append(img_a)
            self.targets.append(0)
            self.data.append(img_b)
            self.targets.append(1)
            cnt+=2
            if(cnt % 10000 == 0):
                print(cnt)
        # print(len(A), len(B), cnt, len(self.data), len(self.targets))
        # print(self.data[0].shape, self.targets[0])
        # print(self.data[-1].shape, self.targets[-1])
        

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


transform_train = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def init_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.IMAGENET1K_V1')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    if torch.cuda.is_available():
        model.to('cuda')
    print(model)
    return model


def train(model, dir_A, dir_B):
    LR = 0.01
    EPOCH = 100
    last = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    writer = SummaryWriter(dir_A + '_resnet_result')
    log_dir = dir_A + '_resnet_result'
    
    trainset = MyData(dir_A, dir_B, train=True, transform=transform_train)
    testset = MyData(dir_A, dir_B, train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

    for epoch in range(EPOCH):
        model.train()
        sum_loss = 0
        correct = 0
        total = 0
        for i, (input_batch, label_batch) in enumerate(train_loader):
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                label_batch = label_batch.to('cuda')
            optimizer.zero_grad()

            outputs = model(input_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label_batch.size(0)
            correct += predicted.eq(label_batch.data).cpu().sum()
            print("[epoch:{}, iter:{}] Loss:{} | Acc: {}".format(epoch+1, i+1+epoch, sum_loss/(i + 1), 100 * correct / total))


        if (epoch + 1) % 10 == 0:
            print("waiting Test...")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in test_loader:
                    model.eval()
                    imgs, lbls = data
                    if torch.cuda.is_available():
                        imgs, lbls = imgs.to('cuda'), lbls.to('cuda')
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += lbls.size(0)
                    correct += (predicted == lbls).sum()
                print("[Test Acc: {}".format(100 * torch.true_divide(correct, total)))
            now = (100 * correct / total).item()
            writer.add_scalars('acc', {'test_acc': now}, epoch)
            if now > last:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state, log_dir+str(int(now))+'.pth')
                last = now

    
def split_txt(dir_A, dir_B):
    if not os.path.exists(dir_A):
        raise ValueError("not exist", dir_A)
    if not os.path.exists(dir_B):
        raise ValueError("not exist", dir_B)
    lis_a = os.listdir(dir_A)
    lis_b = os.listdir(dir_B)

    random.shuffle(lis_a)
    random.shuffle(lis_b)

    a_l = int(len(lis_a) * 0.8)
    b_l = int(len(lis_b) * 0.8)
    train_a = lis_a[:a_l]
    test_a = lis_a[a_l:]
    train_b = lis_b[:b_l]
    test_b = lis_b[b_l:]

    def save_txt(lis_a, lis_b, mode="train"):
        max_n = max(len(lis_a), len(lis_b))
        while 2 * len(lis_a) < max_n:
            lis_a += lis_a
        if len(lis_a) < max_n:
            lis_a += lis_a[:(max_n - len(lis_a))]
        while 2 * len(lis_b) < max_n:
            lis_b += lis_b
        if len(lis_b) < max_n:
            lis_b += lis_b[:(max_n - len(lis_b))]
        
        with open(dir_A + "_" + mode + ".txt", "w") as f:
            for line in lis_a:
                f.write(line + "\n")
        with open(dir_B + "_" + mode + ".txt", "w") as f:
            for line in lis_b:
                f.write(line + "\n")

    save_txt(train_a, train_b, "train")
    save_txt(test_a, test_b, "test")
    


if __name__ == "__main__":
    model = init_model()
    # print(model)
    # train_A = "/home/qing.xiang/algorithm/SophonAlgoNN/train/chefhat"
    # train_B = "/home/qing.xiang/algorithm/SophonAlgoNN/train/unchefhat"
    train_A = "/home/sdb1/xq/algorithm/SophonAlgoNN/data/chefhat"
    train_B = "/home/sdb1/xq/algorithm/SophonAlgoNN/data/unchefhat"
    
    if not os.path.exists(train_A + "_train.txt") or not os.path.exists(train_B + "_train.txt"):
        split_txt(train_A, train_B)
    train(model, train_A, train_B)