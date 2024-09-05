import matplotlib
matplotlib.use('TkAgg')  # Устанавливаем интерактивный бэкенд
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


#Шаг 1: Загрузка данных из idx файлов

# Функция для загрузки изображений
def load_images(file_path):
    with open(file_path, 'rb') as f:
        # Пропускаем заголовок
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Читаем данные
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# Функция для загрузки меток
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        # Пропускаем заголовок
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Читаем данные
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Загрузка обучающих данных
train_images = load_images('data/train-images.idx3-ubyte')
train_labels = load_labels('data/train-labels.idx1-ubyte')

# Загрузка тестовых данных
test_images = load_images('data/t10k-images.idx3-ubyte')
test_labels = load_labels('data/t10k-labels.idx1-ubyte')


# Шаг 2: Создание кастомного датасета

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Шаг 3: Преобразование и загрузка данных

# Преобразование: приводим изображения к формату PyTorch и нормализуем
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Создаем датасеты
train_dataset = MNISTDataset(train_images, train_labels, transform=transform)
test_dataset = MNISTDataset(test_images, test_labels, transform=transform)

# Создаем DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Шаг 4: Обучение и тестирование модели

# Определение модели, функции потерь и оптимизатора остается таким же
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)  # Dropout для регуляризации

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Применяем Dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Применяем Dropout
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Списки для хранения потерь и точности
train_losses = []
test_losses = []
accuracies = []

# Обучение модели
num_epochs = 5
best_accuracy = 0.0
best_test_loss = float('inf')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}')
    

    # Оценка на тестовом наборе
    model.eval()  # Переключаем модель в режим оценки
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Отключаем градиенты для ускорения вычислений
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Вычисляем потери
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Вычисляем точность
            _, predicted = torch.max(outputs.data, 1)  # Определяем предсказанный класс
            total += labels.size(0)  # Общее количество образцов
            correct += (predicted == labels).sum().item()  # Количество правильных предсказаний

    # Средние потери на тестовом наборе
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Точность на тестовом наборе
    accuracy = 100 * correct / total

    # Сохраняем модель только если точность улучшилась
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print('New best model saved!')    

    # # Или, как вариант:
    # # Сохраняем модель только если тестовые потери уменьшились
    # if avg_test_loss < best_test_loss:
    #     best_test_loss = avg_test_loss
    #     torch.save(model.state_dict(), 'best_loss_model.pth')
    #     print('New best model saved based on test loss!')

    # Печать результатов
    print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Обновляем шаг обучения в зависимости от тестовых потерь
    scheduler.step(avg_test_loss)

# Визуализация потерь и точности
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test Accuracy')

plt.show()
