import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# Загрузка данных
df = pd.read_csv('average_temperature_dataset/average-temperature-1900-2023.csv')

# Проверка первых строк данных
print(df.head())

# Примерное содержание датасета:
#    Year  Average_Fahrenheit_Temperature
# 0  1900          14.0593
# 1  1901          14.0633
# 2  1902          14.0567

# Убираем ненужные столбцы и оставляем только "Year" и "AverageTemperature"
data = df['Average_Fahrenheit_Temperature'].values.astype(float)

# Масштабируем данные в диапазон [0, 1] для лучшего обучения RNN
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Визуализация данных
plt.plot(df['Year'], data, label='Average Fahrenheit Temperature')
plt.title("Average Global Temperature from 1900 to 2023")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.legend()
plt.show()

# Функция для создания обучающих последовательностей
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# Параметры
seq_length = 10  # Длина последовательности (10 лет)
x, y = create_sequences(data_scaled, seq_length)

# Превращаем в тензоры для PyTorch
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

# Создаем DataLoader для батчевого обучения
class TemperatureDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = TemperatureDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Определение модели
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Начальное скрытое состояние
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Последнее скрытое состояние для предсказания
        return out

# Параметры модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel().to(device)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Шаг обучения модели
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Прямой проход
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return train_losses

# Обучаем модель
train_losses = train_model(model, train_loader, criterion, optimizer)

# Прогнозирование на обучающих данных
model.eval()
predictions = []
with torch.no_grad():
    for seq, _ in train_loader:
        seq = seq.to(device)
        preds = model(seq).cpu().numpy()
        predictions.extend(preds)

# Визуализация предсказаний
plt.plot(data[seq_length:], label="Actual Temperature")
plt.plot(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)), label="Predicted Temperature")
plt.legend()
plt.title("Actual vs Predicted Global Temperature")
plt.show()

# Логгирование данных для TensorBoard
writer = SummaryWriter()

# Примерные данные для входа (один batch)
sample_input = torch.randn(1, seq_length, 1).to(device)

# Записываем модель в TensorBoard
writer.add_graph(model, sample_input)
writer.close()