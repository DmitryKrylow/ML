from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

img_folder = "C:\\Users\\dima2\\PycharmProjects\\lab6ML"

model = load_model('cifar10_subset_cnn.h5')
classes = ['bird','cat','dog','frog']

def predict_image(path):
    files = sorted(os.listdir(path))

    for file_name in files:
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            # Загружаем как RGB
            img = Image.open(f"{path}\\{file_name}").convert('RGB')
            # Ресайз до 32x32
            img = img.resize((32,32))
            # Преобразуем в массив
            img_array = np.array(img)/255.0
            # Проверка: если изображение всё ещё не 3 канала
            if img_array.shape != (32,32,3):
                print(f"Неверный размер изображения: {img_array.shape}")
                return
            # Добавляем размер батча
            img_array = img_array.reshape(1,32,32,3)
            # Предсказание
            pred = model.predict(img_array)
            predicted_class = classes[np.argmax(pred)]
            print(f"Файл: {file_name} Предсказанный класс: {predicted_class}, вероятности: {[f"{classes[i]}: {(pred[0][i] * 100.0):.4f}%" for i in range(len(classes))] }")


# Пример
predict_image(img_folder)