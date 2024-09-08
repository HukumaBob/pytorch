# Шаг 1: Импорт необходимых библиотек
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt



# Шаг 2: Подготовка данных
# Создадим трансформации для предварительной обработки изображений:
# изменение размера, обрезка и нормализация
# Параметры
batch_size = 32
img_size = 224  # Размер изображения, который мы будем подавать в сеть

# Взято из набора данных ImageNet:
means = [0.485, 0.456, 0.406] 
stds = [0.229, 0.224, 0.225]

# Трансформации для обучающих данных (data augmentation) и для тестовых данных (без изменений)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ]),
}

# Путь к директории dataset
dataset_path = Path('dataset')

# Создание загрузчиков данных для тренировочного и тестового наборов
image_datasets = {
    'train': datasets.ImageFolder(dataset_path / 'train_set', data_transforms['train']),
    'test': datasets.ImageFolder(dataset_path / 'test_set', data_transforms['test'])
}

# Загрузчики данных
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)
}

# Количество классов
num_classes = len(image_datasets['train'].classes)

# # Шаг 3: Создание модели на основе предобученной модели (например, ResNet18)
# # Мы используем ResNet18, предварительно обученную на ImageNet, 
# # и заменим последний полносвязный слой для классификации кошек и собак.
# # Используем предобученную модель ResNet18

# model = models.resnet18(pretrained=True)

# # # Заменим последний полносвязный слой

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, num_classes)  # num_classes = 2 (cats, dogs)

# Или обучим свою сеть сами:
# Создание собственной CNN-модели
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Сверточный слой 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Сверточный слой 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Сверточный слой 3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Пуллинг (макспулинг)
        self.fc1 = nn.Linear(128 * (img_size // 8) * (img_size // 8), 512)  # Полносвязный слой
        self.fc2 = nn.Linear(512, num_classes)  # Выходной слой (2 класса: кошки и собаки)
        self.relu = nn.ReLU()  # Активация ReLU
        self.dropout = nn.Dropout(0.5)  # Dropout для предотвращения переобучения

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Сверточный слой 1 -> ReLU -> Пуллинг
        x = self.pool(self.relu(self.conv2(x)))  # Сверточный слой 2 -> ReLU -> Пуллинг
        x = self.pool(self.relu(self.conv3(x)))  # Сверточный слой 3 -> ReLU -> Пуллинг
        x = x.view(-1, 128 * (img_size // 8) * (img_size // 8))  # Выровнять (flatten)
        x = self.dropout(self.relu(self.fc1(x)))  # Полносвязный слой 1 -> ReLU -> Dropout
        x = self.fc2(x)  # Выходной слой
        return x

# Создаем модель
model = SimpleCNN(num_classes=num_classes)
# Использование GPU, если доступно
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Шаг 4: Определение функции потерь и оптимизатора
# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Шаг 5: Функция обучения
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    model.train()  # Устанавливаем модель в режим обучения

    train_losses = []  # Для сохранения потерь на каждой эпохе
    train_accuracies = []  # Для сохранения точности на каждой эпохе    

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Обратный проход
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        accuracy = correct / total

        train_losses.append(epoch_loss)  # Сохраняем потери
        train_accuracies.append(accuracy)  # Сохраняем точность    

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return train_losses, train_accuracies

# Шаг 6: Тестирование модели
# После обучения проверим модель на тестовых данных.
def evaluate_model(model, dataloaders):
    model.eval()  # Устанавливаем модель в режим оценки
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test']):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Шаг 7: Запуск обучения и тестирования
num_epochs = 10
train_losses, train_accuracies = train_model(model, dataloaders, criterion, optimizer, num_epochs)
test_accuracy = evaluate_model(model, dataloaders)

# Построение графиков
epochs = range(1, num_epochs + 1)

# График потерь
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# График точности
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'ro-', label='Training accuracy')
plt.axhline(y=test_accuracy, color='g', linestyle='--', label=f'Test Accuracy: {test_accuracy:.4f}')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

