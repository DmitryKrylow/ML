from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

# Загружаем CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Классы: ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
selected_classes = [2, 3, 5, 6]  # bird, cat, dog, frog

y_train = y_train.flatten()
y_test = y_test.flatten()

# Фильтруем обучающие данные
train_mask = np.isin(y_train, selected_classes)
x_train = x_train[train_mask]
y_train = y_train[train_mask]

# Фильтруем тестовые данные
test_mask = np.isin(y_test, selected_classes)
x_test = x_test[test_mask]
y_test = y_test[test_mask]

# Перенумеровываем метки на 0,1,2,3
class_mapping = {2:0, 3:1, 5:2, 6:3}
y_train = np.array([class_mapping[y] for y in y_train])
y_test = np.array([class_mapping[y] for y in y_test])

# One-hot encoding
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# Нормализуем изображения
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 класса
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# Сохраняем модель

model.save('cifar10_subset_cnn.h5')
