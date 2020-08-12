# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from student_net import *
"""
# Load進我們的Model架構(在hw7_Architecture_Design.ipynb內)
!gdown --id '1lJS0ApIyi7eZ2b3GMyGxjPShI8jXM2UC' --output "hw7_Architecture_Design.ipynb"
# %run "hw7_Architecture_Design.ipynb"
"""

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

"""# Data Processing

我們的Dataset使用的是跟Hw3 - CNN同樣的Dataset，因此這個區塊的Augmentation / Read Image大家參考或直接抄就好。

如果有不會的話可以回去看Hw3的colab。

需要注意的是如果要自己寫的話，Augment的方法最好使用我們的方法，避免輸入有差異導致Teacher Net預測不好。
"""

import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0

            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        f'../dataset/food-11/{mode}',
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader

"""# Pre-processing

我們已經提供TeacherNet的state_dict，其架構是torchvision提供的ResNet18。

至於StudentNet的架構則在hw7_Architecture_Design.ipynb中。

這裡我們使用的Optimizer為AdamW，沒有為甚麼，就純粹我想用。
"""

# get dataloader
train_dataloader = get_dataloader('training', batch_size=32)
valid_dataloader = get_dataloader('validation', batch_size=32)

teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
student_net = StudentNet(base=16).cuda()

teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)

"""# Start Training

* 剩下的步驟與你在做Hw3 - CNN的時候一樣。

## 小提醒

* torch.no_grad是指接下來的運算或該tensor不需要算gradient。
* model.eval()與model.train()差在於Batchnorm要不要紀錄，以及要不要做Dropout。
"""

def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


# TeacherNet永遠都是Eval mode.
teacher_net.eval()
now_best_acc = 0
for epoch in range(200):
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model.bin')
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))

"""# Inference

同Hw3，請參考該作業:)。

# Q&A

有任何問題Network Compression的問題可以寄信到b05902127@ntu.edu.tw / ntu-ml-2020spring-ta@googlegroups.com。

時間允許的話我會更新在這裡。
"""
