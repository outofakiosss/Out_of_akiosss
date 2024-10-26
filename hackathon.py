import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import json
import os

# Пути к данным
weights_path = 'C:/Users/1/Desktop/best_handwritten_classifier.pth'
output_json_path = 'C:/Users/1/Desktop/output.json'


# --- 1. Модель для классификации отдельных символов ---
class CharacterClassifierCNN(nn.Module):
    """CNN для классификации отдельных букв"""

    def __init__(self):
        super(CharacterClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 33)  # 33 буквы русского алфавита + 1 класс для пробела

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. Загрузка обученной модели ---
classifier = CharacterClassifierCNN()
if os.path.exists(weights_path):
    classifier.load_state_dict(torch.load(weights_path))
    classifier.eval()


# --- 3. Функция сегментации рукописного текста на изображении ---
def segment_text_blocks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10 and w > 10:  # Игнорировать мелкие объекты
            text_blocks.append(image[y:y + h, x:x + w])
    return text_blocks


# --- 4. Функция распознавания букв ---
def detect_and_recognize_characters(image, output_json_path):
    text_blocks = segment_text_blocks(image)
    recognition_results = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # Сопоставление индексов с русскими буквами
    alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "

    for block in text_blocks:
        block_tensor = transform(block).unsqueeze(0)
        with torch.no_grad():
            output = classifier(block_tensor)
            _, predicted = torch.max(output, 1)

        recognized_character = alphabet[predicted.item()]
        recognition_results.append({"content": recognized_character, "signature": False})

    save_recognition_to_json(recognition_results, output_json_path)


# --- 5. Функция сохранения результатов в JSON ---
def save_recognition_to_json(results, output_path):
    data = [{"content": result["content"], "signature": result["signature"]} for result in results]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# --- Пример использования ---
image_path = 'C:/Users/1/Desktop/0.jpg'
image = cv2.imread(image_path)
detect_and_recognize_characters(image, output_json_path)
print(f"Результаты сохранены в {output_json_path}")
