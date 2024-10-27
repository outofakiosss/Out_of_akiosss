import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Параметры
weights_path = '/home/user/pyenv/desktop/toserver/best_handwritten_classifier.pth'
folder_path = '/home/user/pyenv/desktop/archive_1'  # Корневая папка с изображениями
output_json_folder = "/home/user/pyenv/desktop/toserver/"  # Папка для сохранения JSON-результатов
num_epochs = 20
batch_size = 64
learning_rate = 0.001

# Словарь меток
label_to_letter = {
    0: "а", 1: "б", 2: "в", 3: "г", 4: "д", 5: "е", 6: "ё", 7: "ж", 8: "з", 9: "и",
    10: "й", 11: "к", 12: "л", 13: "м", 14: "н", 15: "о", 16: "п", 17: "р", 18: "с",
    19: "т", 20: "у", 21: "ф", 22: "х", 23: "ц", 24: "ч", 25: "ш", 26: "щ", 27: "ъ",
    28: "ы", 29: "ь", 30: "э", 31: "ю", 32: "я", 33: "А", 34: "Б", 35: "В", 36: "Г",
    37: "Д", 38: "Е", 39: "Ё", 40: "Ж", 41: "З", 42: "И", 43: "Й", 44: "К", 45: "Л",
    46: "М", 47: "Н", 48: "О", 49: "П", 50: "Р", 51: "С", 52: "Т", 53: "У", 54: "Ф",
    55: "Х", 56: "Ц", 57: "Ч", 58: "Ш", 59: "Щ", 60: "Ъ", 61: "Ы", 62: "Ь", 63: "Э",
    64: "Ю", 65: "Я"
}

# Определение порядка папок
folder_names = [
    "00_00_00", "00_01_00", "00_02_00", "00_03_00", "00_04_00", "00_05_00", "00_06_00", "00_07_00", "00_08_00",
    "00_09_00", "00_10_00", "00_11_00", "00_12_00", "00_13_00", "00_14_00", "00_15_00", "00_16_00", "00_17_00",
    "00_18_00", "00_19_00", "00_20_00", "00_21_00", "00_22_00", "00_23_00", "00_24_00", "00_25_00", "00_26_00",
    "00_27_00", "00_28_00", "00_29_00", "00_30_00", "00_31_00", "00_32_00", "01_00_00", "01_01_00", "01_02_00",
    "01_03_00", "01_04_00", "01_05_00", "01_06_00", "01_07_00", "01_08_00", "01_09_00", "01_10_00", "01_11_00",
    "01_12_00", "01_13_00", "01_14_00", "01_15_00", "01_16_00", "01_17_00", "01_18_00", "01_19_00", "01_20_00",
    "01_21_00", "01_22_00", "01_23_00", "01_24_00", "01_25_00", "01_26_00", "01_27_00", "01_28_00", "01_29_00",
    "01_30_00", "01_31_00", "01_32_00"
]

folder_to_label = {folder: idx for idx, folder in enumerate(folder_names)}

# Модель CNN
class HandwrittenClassifierCNN(nn.Module):
    def __init__(self):
        super(HandwrittenClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 66)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandwrittenClassifierCNN().to(device)
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
else:
    print("Файл с весами не найден. Будет проведено обучение модели с нуля.")

# Подготовка изображений с рекурсивным обходом
def find_images_recursively(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Определение CustomDataset с предобработкой изображений
class CustomDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        processed_image = self.transform(binary)
        label = self.get_label_from_path(img_path)
        return processed_image, label

    def get_label_from_path(self, path):
        folder_name = os.path.basename(os.path.dirname(path))
        if folder_name in folder_to_label:
            return folder_to_label[folder_name]
        else:
            raise ValueError(f"Неизвестное имя папки {folder_name}. Проверьте folder_to_label.")

# Настройка DataLoader
images = find_images_recursively(folder_path)
train_dataset = CustomDataset(images)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Функция для распознавания символов
def detect_and_recognize_characters(image_path, output_json_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10 and w > 10:
            text_block = image[y:y+h, x:x+w]
            processed_image = transforms.ToTensor()(text_block).unsqueeze(0).to(device)
            output = model(processed_image)
            _, predicted = torch.max(output, 1)
            recognized_letter = label_to_letter[predicted.item()]
            results.append({"position": (x, y, w, h), "letter": recognized_letter})

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

# Обработка всех изображений и сохранение JSON результатов
if not os.path.exists(output_json_folder):
    os.makedirs(output_json_folder)

for image_path in images:
    output_json_path = os.path.join(output_json_folder, f"{os.path.basename(image_path)}.json")
    detect_and_recognize_characters(image_path, output_json_path)