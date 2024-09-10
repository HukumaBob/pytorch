import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Создание искусственного датасета
class SequenceDataset(Dataset):
    def __init__(self, num_sequences, seq_length, input_size):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.input_size = input_size
        self.data, self.labels = self._generate_sequences()

    def _generate_sequences(self):
        # Генерация последовательностей случайных чисел
        data = np.random.rand(self.num_sequences, self.seq_length, self.input_size)
        labels = (np.mean(data, axis=1) > 0.5).astype(int)  # Классификация на основе среднего значения
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Параметры данных
num_sequences = 1000  # Количество последовательностей
seq_length = 10       # Длина каждой последовательности
input_size = 1        # Размерность каждого временного шага

train_dataset = SequenceDataset(num_sequences, seq_length, input_size)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Шаг 2: Определение модели RNN
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN слой
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Полносвязный слой для классификации
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Инициализация начального скрытого состояния
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Прямой проход через RNN
        out, _ = self.rnn(x, h0)
        
        # Используем выход последнего временного шага
        out = self.fc(out[:, -1, :])
        return out

# Параметры модели
input_size = 1       # Входная размерность (одно число на временной шаг)
hidden_size = 32     # Количество скрытых нейронов
num_layers = 2       # Количество слоёв RNN
num_classes = 2      # Два класса (0 и 1)

model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)

# Шаг 3: Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Шаг 4: Обучение модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()

    # Списки для сохранения потерь и точности
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            labels = labels.squeeze()

            optimizer.zero_grad()

            # Прямой проход
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Обратный проход
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        accuracy = correct / total

        # Сохраняем потери и точность для каждой эпохи
        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

    return train_losses, train_accuracies

def plot_training_results(num_epochs, train_losses, train_accuracies):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'ro-', label='Training accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Шаг 5: Тренировка
num_epochs = 20
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Шаг 6: Тестирование модели (на обучающих данных)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f'Test Accuracy: {correct / total:.4f}')

train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs)
plot_training_results(num_epochs, train_losses, train_accuracies)
