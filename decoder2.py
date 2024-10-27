import os
import json
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms

# Параметры
yolo_model_path = '/home/user/pyenv/desktop/toserver/yolo_model.pt'  # Замените на реальный путь к модели YOLO
weights_path = '/home/user/pyenv/desktop/toserver/best_handwritten_classifier.pth'
output_json_dir = '/path/to/output'  # Замените на нужный путь

# Загрузка модели YOLO
yolo_model = YOLO(yolo_model_path)

# Загрузка модели классификатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = HandwrittenClassifierCNN().to(device)
classifier.load_state_dict(torch.load(weights_path, map_location=device))
classifier.eval()

# Предобработка изображений
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Функция для обнаружения и распознавания символов
def detect_and_recognize_characters(image_path, output_json_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)  # Используем YOLO для обнаружения объектов

    # Получаем координаты для каждого обнаруженного объекта
    detections = results.pred[0]  # Получаем предсказания
    recognized_results = []

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        if conf > 0.5:  # Фильтр по уверенности
            text_block = image[y1:y2, x1:x2]
            processed_image = transform(text_block).unsqueeze(0).to(device)
            output = classifier(processed_image)
            _, predicted = torch.max(output, 1)
            recognized_letter = label_to_letter[predicted.item()]

            recognized_results.append({
                "position": (x1, y1, x2, y2),
                "confidence": conf.item(),
                "letter": recognized_letter
            })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(recognized_results, f, ensure_ascii=False)

# Пример использования
folder_path = '/home/user/pyenv/desktop/toserver'  # Замените на ваш путь к папке
image_paths = find_images_recursively(folder_path)

# Обработка каждого изображения
for image_path in image_paths:
    output_json_path = os.path.join(output_json_dir, f"{os.path.basename(image_path)}.json")
    detect_and_recognize_characters(image_path, output_json_path)
