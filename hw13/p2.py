# -*- coding: utf-8 -*-

## **Step 1: 下載資料**

from PIL import Image
"""
from IPython.display import display
for i in range(10, 20):
  im = Image.open("Omniglot/images_background/Japanese_(hiragana).0/character13/0500_" + str (i) + ".png")
  display(im)

"""
## **Step 2: 建立模型**"""

# Import modules we need
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import glob
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

def ConvBlock(in_ch, out_ch):
  return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding = 1),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size = 2, stride = 2)) 

def ConvBlockFunction(x, w, b, w_bn, b_bn):
  x = F.conv2d(x, w, b, padding = 1)
  x = F.batch_norm(x, running_mean = None, running_var = None, weight = w_bn, bias = b_bn, training = True)
  x = F.relu(x)
  x = F.max_pool2d(x, kernel_size = 2, stride = 2)
  return x

class Classifier(nn.Module):
  def __init__(self, in_ch, k_way):
    super(Classifier, self).__init__()
    self.conv1 = ConvBlock(in_ch, 64)
    self.conv2 = ConvBlock(64, 64)
    self.conv3 = ConvBlock(64, 64)
    self.conv4 = ConvBlock(64, 64)
    self.logits = nn.Linear(64, k_way)
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(x.shape[0], -1)
    x = self.logits(x)
    return x

  def functional_forward(self, x, params):
    '''
    Arguments:
    x: input images [batch, 1, 28, 28]
    '''
    for block in [1, 2, 3, 4]:
      x = ConvBlockFunction(x, params[f'conv{block}.0.weight'], params[f'conv{block}.0.bias'],
                            params.get(f'conv{block}.1.weight'), params.get(f'conv{block}.1.bias'))
    x = x.view(x.shape[0], -1)
    x = F.linear(x, params['logits.weight'] , params['logits.bias'])
    return x


def create_label(n_way, k_shot):
  return torch.arange(n_way).repeat_interleave(k_shot).long()
  

def MAML(model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step = 1, inner_lr = 0.4, train = True):
  """
  Args:
  x is the input omniglot images for a meta_step, shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
  n_way: num of class in each task
  k_shot: num of images in each class when training
 _query: num of images to update in each class when testing
  """
  criterion = loss_fn
  task_loss = []  
  task_acc = []   
  for meta_batch in x:
    train_set = meta_batch[:n_way*k_shot] 
    val_set = meta_batch[n_way*k_shot:]   
    
    fast_weights = OrderedDict(model.named_parameters()) 
    g = []
    eps = 0.7
    for name, param in fast_weights.items():
        g.append(torch.zeros_like(param))
    for inner_step in range(inner_train_step): 
      train_label = create_label(n_way, k_shot).cuda()
      logits = model.functional_forward(train_set, fast_weights)
      loss = criterion(logits, train_label)
      grads = torch.autograd.grad(loss, fast_weights.values(), create_graph = True)
      for i in range(len(g)):
          g[i] += grads[i] ** 2
      fast_weights = OrderedDict((name, param - inner_lr * grad / torch.sqrt(_g + eps))
                                  for ((name, param), grad, _g) in zip(fast_weights.items(), grads, g))
  
    val_label = create_label(n_way, q_query).cuda()
    logits = model.functional_forward(val_set, fast_weights) 
    loss = criterion(logits, val_label)                 
    task_loss.append(loss)                          
    acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean() 
    task_acc.append(acc)
    
  model.train()
  optimizer.zero_grad()
  meta_batch_loss = torch.stack(task_loss).mean()
  if train:
    meta_batch_loss.backward()
    optimizer.step()
  task_acc = np.mean(task_acc)
  return meta_batch_loss, task_acc


class Omniglot(Dataset):
  def __init__(self, data_dir, k_way, q_query):
    self.file_list = [f for f in glob.glob(data_dir + "**/character*", recursive=True)]
    self.transform = transforms.Compose([transforms.ToTensor()])
    self.n = k_way + q_query
  def __getitem__(self, idx):
    sample = np.arange(20)
    np.random.shuffle(sample) 
    img_path = self.file_list[idx]
    img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
    img_list.sort()
    imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
    imgs = torch.stack(imgs)[sample[:self.n]]
    return imgs
  def __len__(self):
    return len(self.file_list)

"""## **Step 3: hyperparameter
"""

n_way = 5
k_shot = 1
q_query = 1
inner_train_step = 1
inner_lr = 0.4
meta_lr = 0.001
meta_batch_size = 32
max_epoch = 40
eval_batches = test_batches = 20
train_data_path = './Omniglot/images_background/'
test_data_path = './Omniglot/images_evaluation/'

train_set, val_set = torch.utils.data.random_split(Omniglot(train_data_path, k_shot, q_query), [3200,656])
train_loader = DataLoader(train_set,
                          batch_size = n_way, 
                          num_workers = 8,
                          shuffle = True,
                          drop_last = True)
val_loader = DataLoader(val_set,
                          batch_size = n_way,
                          num_workers = 8,
                          shuffle = True,
                          drop_last = True)
test_loader = DataLoader(Omniglot(test_data_path, k_shot, q_query),
                          batch_size = n_way,
                          num_workers = 8,
                          shuffle = True,
                          drop_last = True)
train_iter = iter(train_loader)
val_iter = iter(val_loader)
test_iter = iter(test_loader)

meta_model = Classifier(1, n_way).cuda()
optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
loss_fn = nn.CrossEntropyLoss().cuda()


def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
  data = []
  for _ in range(meta_batch_size):
    try:
      task_data = iterator.next()  
    except StopIteration:
      iterator = iter(data_loader)
      task_data = iterator.next()
    train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
    val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
    task_data = torch.cat((train_data, val_data), 0)
    data.append(task_data)
  return torch.stack(data).cuda(), iterator


for epoch in range(max_epoch):
  print("Epoch %d" %(epoch))
  train_meta_loss = []
  train_acc = []
  for step in tqdm(range(len(train_loader) // (meta_batch_size))): 
    x, train_iter = get_meta_batch(meta_batch_size, k_shot, q_query, train_loader, train_iter)
    meta_loss, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step = 5)
    train_meta_loss.append(meta_loss.item())
    train_acc.append(acc)
  print("  Loss    : ", np.mean(train_meta_loss))
  print("  Accuracy: ", np.mean(train_acc))

  val_acc = []
  for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
    x, val_iter = get_meta_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
    _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step = 3, train = False) 
    val_acc.append(acc)
  print("  Validation accuracy: ", np.mean(val_acc))


test_acc = []
for test_step in tqdm(range(len(test_loader) // (test_batches))):
  x, test_iter = get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
  _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step = 3, train = False) 
  test_acc.append(acc)
print("  Testing accuracy: ", np.mean(test_acc))
