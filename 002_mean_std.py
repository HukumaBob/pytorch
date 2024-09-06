import torch
from torchvision import datasets, transforms

# Загрузка набора данных MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Стек всех изображений из набора данных (для удобства вычисления)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
data = next(iter(train_loader))[0]

# Вычисление среднего и стандартного отклонения для всех изображений
mean = data.mean().item()  # Среднее значение всех пикселей
std = data.std().item()    # Стандартное отклонение всех пикселей

print(f"Mean: {mean}, Std: {std}")
