# torch — основной модуль для работы с тензорами и выполнения вычислений на GPU.
# nn — модуль для создания нейронных сетей.
# DataLoader — объект для загрузки данных по мини-пакетам (batch).
# datasets — модуль для работы с популярными датасетами.
# ToTensor — трансформация, которая преобразует изображения в тензоры.
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
# Используются встроенные датасеты FashionMNIST (изображения одежды).
# Данные загружаются в виде тренировочного набора (train=True) и тестового набора (train=False).
# Применяется трансформация ToTensor(), которая конвертирует изображения в тензоры PyTorch.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Размер батча (mini-batch) устанавливается равным 64.
# DataLoader используется для итеративной загрузки данных по мини-пакетам.
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Один пакет данных загружается из test_dataloader.
# Выводятся размеры тензоров X (изображения) и y (метки классов).
# Формат X — [N, C, H, W], где:

#     N — количество изображений в батче,
#     C — число каналов (1 для черно-белых изображений),
#     H, W — высота и ширина изображения.
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
# Определяется, использовать ли GPU (CUDA), MPS (Apple Silicon) или CPU для вычислений.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
# Определяется класс нейронной сети NeuralNetwork, наследующий nn.Module.
# Архитектура состоит из:

#     Операции Flatten(), которая преобразует изображения размером 28x28 в вектор длиной 784 (28*28).
#     Трех полносвязных слоев Linear с активацией ReLU.
#     Последний слой возвращает 10 выходных значений (по числу классов).
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Создается объект модели и перемещается на выбранное устройство (CPU или GPU).
model = NeuralNetwork().to(device)
print(model)

# Функция потерь: CrossEntropyLoss используется для многоклассовой классификации.
# Оптимизатор: стохастический градиентный спуск (SGD) с шагом обучения 1e-3.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Функция train выполняет обучение:
#     Переход модели в режим обучения (model.train()).
#     Для каждого батча данных:
#         Перемещение данных на устройство.
#         Прогон батча через модель.
#         Вычисление функции потерь.
#         Обратное распространение ошибки.
#         Шаг оптимизации.
#         Сброс градиентов.
#     Каждые 100 батчей выводится текущая потеря.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Функция test выполняет проверку:
#     Переход в режим оценки (model.eval()).
#     Отключение градиентов с помощью torch.no_grad() для ускорения вычислений.
#     Подсчет общей ошибки и точности модели.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Модель обучается 5 эпох. После каждой эпохи вызывается функция обучения и тестирования.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Модель сохраняется в файл model.pth.
# После этого модель может быть загружена и восстановлена.
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# Выполняется предсказание для первого изображения из тестового набора.
# Выводится предсказанный и реальный класс.
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')