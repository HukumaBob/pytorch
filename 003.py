import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

trans = tv.transforms.Compose(
    [tv.transforms.ToTensor()]
)

ds_mnist = tv.datasets.MNIST('./datasets', download=True, transform=trans)

# ds_mnist[100][0].numpy()[0].shape
# plt.imshow(ds_mnist[100][0].numpy()[0])

batch_size = 16
dataloader = torch.utils.data.DataLoader(
    ds_mnist, batch_size=batch_size,
    shuffle=True, num_workers=1, drop_last=True
    )

class Neural_number(nn.Module):
  def __init__(self):
    super().__init__()
    self.flat = nn.Flatten()
    self.linear1 = nn.Linear(28*28, 100)
    self.linear2 = nn.Linear(100, 10)
    self.act = nn.ReLU()

  def forward(self, x):
    out = self.flat(x)
    out = self.linear1(out)
    out = self.act(out)
    out = self.linear2(out)

    return out
  
def count_parameters(model):
   return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = Neural_number()

print('count_parameters: ', count_parameters(model))

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

def accuracy(pred, label):
   answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
   return answer.mean()

epochs = 5

for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for img, label in (pbar := tqdm(dataloader)):
        optimizer.zero_grad() 

        label = F.one_hot(label, 10).float()
        pred = model(img)

        loss = loss_fn(pred, label)
        
        loss.backward()
        loss_item = loss.item()
        loss_val += loss_item

        optimizer.step()

        acc_current = accuracy(pred, label)
        acc_val += acc_current
        pbar.set_description(f'loss: {loss_item: .5f}\taccuracy: {acc_current: .3f}')

    print(loss_val/len(dataloader))
    print(acc_val/len(dataloader))

    img = cv2.imread('007.png', cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)/255.0

    t_img = torch.from_numpy(img)
    real_pred = model(t_img)

    answ = F.softmax(real_pred).detach().numpy().argmax()

    print(f'this is: {answ}')