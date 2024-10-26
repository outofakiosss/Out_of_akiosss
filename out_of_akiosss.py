# Импортируем необходимые библиотеки
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.utils import shuffle

# Пути к файлам с весами и к папкам с изображениями
weights_path = 'C:/Users/1/Desktop/best_handwritten_classifier.pth'

# Массив папок для каждой буквы алфавита
folders = [
    'C:/Users/1/Desktop/archive_1/00_00_00/00_00_00',  # Папка с изображениями буквы "а"
    'C:/Users/1/Desktop/archive_1/00_01_00/00_01_00',  # Папка с изображениями буквы "б"
    'C:/Users/1/Desktop/archive_1/00_02_00/00_02_00',  # Папка с изображениями буквы "в"
    'C:/Users/1/Desktop/archive_1/00_03_00/00_03_00',  # Папка с изображениями буквы "г"
    'C:/Users/1/Desktop/archive_1/00_04_00/00_04_00',  # Папка с изображениями буквы "д"
    'C:/Users/1/Desktop/archive_1/00_05_00/00_05_00',  # Папка с изображениями буквы "е"
    'C:/Users/1/Desktop/archive_1/00_06_00/00_06_00',  # Папка с изображениями буквы "ё"
    'C:/Users/1/Desktop/archive_1/00_07_00/00_07_00',  # Папка с изображениями буквы "ж"
    'C:/Users/1/Desktop/archive_1/00_08_00/00_08_00',  # Папка с изображениями буквы "з"
    'C:/Users/1/Desktop/archive_1/00_09_00/00_09_00',  # Папка с изображениями буквы "и"
    'C:/Users/1/Desktop/archive_1/00_10_00/00_10_00',  # Папка с изображениями буквы "й"
    'C:/Users/1/Desktop/archive_1/00_11_00/00_11_00',  # Папка с изображениями буквы "к"
    'C:/Users/1/Desktop/archive_1/00_12_00/00_12_00',  # Папка с изображениями буквы "л"
    'C:/Users/1/Desktop/archive_1/00_13_00/00_13_00',  # Папка с изображениями буквы "м"
    'C:/Users/1/Desktop/archive_1/00_14_00/00_14_00',  # Папка с изображениями буквы "н"
    'C:/Users/1/Desktop/archive_1/00_15_00/00_15_00',  # Папка с изображениями буквы "о"
    'C:/Users/1/Desktop/archive_1/00_16_00/00_16_00',  # Папка с изображениями буквы "п"
    'C:/Users/1/Desktop/archive_1/00_17_00/00_17_00',  # Папка с изображениями буквы "р"
    'C:/Users/1/Desktop/archive_1/00_18_00/00_18_00',  # Папка с изображениями буквы "с"
    'C:/Users/1/Desktop/archive_1/00_19_00/00_19_00',  # Папка с изображениями буквы "т"
    'C:/Users/1/Desktop/archive_1/00_20_00/00_20_00',  # Папка с изображениями буквы "у"
    'C:/Users/1/Desktop/archive_1/00_21_00/00_21_00',  # Папка с изображениями буквы "ф"
    'C:/Users/1/Desktop/archive_1/00_22_00/00_22_00',  # Папка с изображениями буквы "х"
    'C:/Users/1/Desktop/archive_1/00_23_00/00_23_00',  # Папка с изображениями буквы "ц"
    'C:/Users/1/Desktop/archive_1/00_24_00/00_24_00',  # Папка с изображениями буквы "ч"
    'C:/Users/1/Desktop/archive_1/00_25_00/00_25_00',  # Папка с изображениями буквы "ш"
    'C:/Users/1/Desktop/archive_1/00_26_00/00_26_00',  # Папка с изображениями буквы "щ"
    'C:/Users/1/Desktop/archive_1/00_27_00/00_27_00',  # Папка с изображениями буквы "ъ"
    'C:/Users/1/Desktop/archive_1/00_28_00/00_28_00',  # Папка с изображениями буквы "ы"
    'C:/Users/1/Desktop/archive_1/00_29_00/00_29_00',  # Папка с изображениями буквы "ь"
    'C:/Users/1/Desktop/archive_1/00_30_00/00_30_00',  # Папка с изображениями буквы "э"
    'C:/Users/1/Desktop/archive_1/00_31_00/00_31_00',  # Папка с изображениями буквы "ю"
    'C:/Users/1/Desktop/archive_1/00_32_00/00_32_00'   # Папка с изображениями буквы "я"
]


# 1. Подготовка датасета
class LettersDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Сегментация областей текста
def segment_text_blocks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10 and w > 10:
            text_blocks.append(image[y:y+h, x:x+w])

    return text_blocks

# Архитектура CNN для классификации текста
class HandwrittenClassifierCNN(nn.Module):
    def __init__(self):
        super(HandwrittenClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 33)  # 33 выхода — по одному для каждой буквы

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Инициализация сети, функции потерь и оптимизатора
model = HandwrittenClassifierCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Проверка и загрузка весов
if os.path.exists(weights_path):
    print("Загружаем сохранённые веса модели...")
    model.load_state_dict(torch.load(weights_path))
    model.eval()
else:
    print("Файл с весами не найден. Начинаем обучение с нуля.")

# Предобработка изображений
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Загрузка изображений и меток для каждой буквы
images = []
labels = []
for idx, folder in enumerate(folders):
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
            labels.append(idx)  # Метка равна индексу буквы в массиве `folders`

# Перемешивание данных
images, labels = shuffle(images, labels, random_state=42)

# Создание DataLoader
dataset = LettersDataset(images, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Обучение модели
num_epochs = 20  # Увеличенное количество эпох
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Эпоха [{epoch+1}/{num_epochs}], Потери: {total_loss/len(dataloader):.4f}")

# Сохранение модели
torch.save(model.state_dict(), "C:/Users/1/Desktop/best_handwritten_classifier.pth")
print("Обучение завершено и модель сохранена!")
