import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

SIZE = 256

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
            if not os.path.exists(path_a):
                raise ValueError("not exist", path_a)
            if not os.path.exists(path_b):
                raise ValueError("not exist", path_b)
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
    transforms.Resize(SIZE),
    transforms.RandomCrop(SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.RandomCrop(SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def init_model(checkpoint=None, cls_num=None):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.IMAGENET1K_V1')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    if cls_num:
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=cls_num)
    if checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model'])
    if torch.cuda.is_available():
        model.to('cuda')
    # for k, v in model.named_parameters():
    #     print(k)
    #     if k.startswith("fc"):
    #         v.requires_grad = True
    #     else:
    #         v.requires_grad = False
    # print(model.fc)
    return model


def train(model, dir_A, dir_B, epoch=40, batch_size=128):
    LR = 0.001
    last = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    writer = SummaryWriter(dir_A + '_resnet_result')
    log_dir = dir_A + '_resnet_result'
    
    trainset = MyData(dir_A, dir_B, train=True, transform=transform_train)
    testset = MyData(dir_A, dir_B, train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    for epoch in range(epoch):
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


        if (epoch + 1) % 2 == 0:
            model.eval()
            print("waiting Test...")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in test_loader:
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


def infer(model, infer_dir):
    model.eval()
    os.makedirs(infer_dir + "_0", exist_ok=True)
    os.makedirs(infer_dir + "_1", exist_ok=True)
    n_cls_0, n_cls_1 = 0, 0
    for img_name in tqdm(sorted(os.listdir(infer_dir))):
        img_path = os.path.join(infer_dir, img_name)
        imgsrc = cv2.imread(img_path)
        img = Image.fromarray(imgsrc)        
        img = transform_test(img)
        img = torch.unsqueeze(img, dim=0)
        if torch.cuda.is_available():
            img = img.to('cuda')
        # print(img.size(), torch.min(img), torch.max(img))
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        # print(outputs)
        if predicted.item() == 0:
            n_cls_0 += 1
            save_path = os.path.join(infer_dir + "_0", img_name)
        elif predicted.item() == 1:
            n_cls_1 += 1
            save_path = os.path.join(infer_dir + "_1", img_name)
        else:
            print("output is", predicted.item())
        cv2.imwrite(save_path, imgsrc)
    print("0: {}, 1: {}".format(n_cls_0, n_cls_1))


def pth2onnx(model, save_path):
    x = torch.randn(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model, 
        x, 
        save_path,
        opset_version=10,
        do_constant_folding=True,	# 是否执行常量折叠优化
        input_names=["input"],	# 输入名
        output_names=["output"],	# 输出名
        dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                      "output":{0:"batch_size"}}
        )



def infer_resnet_pth(model, imgsrc):
    img = Image.fromarray(imgsrc)        
    img = transform_test(img)
    img = torch.unsqueeze(img, dim=0)
    if torch.cuda.is_available():
        img = img.to('cuda')
    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()


def infer_resnet_onnx(onnx_session, imgsrc):
    img = Image.fromarray(imgsrc)        
    img = transform_test(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.detach().cpu().numpy() if img.requires_grad else img.cpu().numpy()
    outputs = onnx_session.run(None, {"input": img})[0]
    outputs = np.array(outputs)
    predicted = np.argmax(outputs, 1)
    return predicted[0]


if __name__ == "__main__":
    # model = init_model(cls_num=2)
    # # train_A = "/home/sdb1/xq/algorithm/SophonAlgoNN/data/nude/cropped_shirtless"
    # # train_B = "/home/sdb1/xq/algorithm/SophonAlgoNN/data/nude/cropped_noshirtless"
    # train_A = "/dataset/user/xq/resnet/chefhat/cropped_chefhat"
    # train_B = "/dataset/user/xq/resnet/chefhat/cropped_nochefhat"
    
    # # if not os.path.exists(train_A + "_train.txt") or not os.path.exists(train_B + "_train.txt"):
    # split_txt(train_A, train_B)
    # # global SIZE
    # SIZE = 128
    # train(model, train_A, train_B, epoch=100, batch_size=128)

    
    # model = init_model("/home/sdb1/xq/algorithm/SophonAlgoNN/data/chefhat_resnet/chefhat_resnet_v7_92.pth", cls_num=2)
    # infer_dir = "/home/sdb1/xq/algorithm/SophonAlgoNN/data/chefhat_resnet/cropped"
    # infer(model, infer_dir)

    model = init_model("/dataset/user/xq/resnet/chefhat/cropped_chefhat_resnet_result98.pth", cls_num=2)
    model.to('cpu')
    save_path = "/dataset/user/xq/resnet/chefhat/chefhat_resnet.onnx"
    pth2onnx(model, save_path)