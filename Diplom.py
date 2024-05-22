import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка и подготовка данных
train_data = pd.read_csv("/kaggle/input/egyptian-hieroglyphs/train/_annotations.csv")
train_data.info()
valid_data = pd.read_csv("/kaggle/input/egyptian-hieroglyphs/valid/_annotations.csv")

test_data = pd.read_csv("/kaggle/input/egyptian-hieroglyphs/test/_annotations.csv")

import os
import cv2

train_folder = "/kaggle/input/egyptian-hieroglyphs/train/"
valid_folder = "/kaggle/input/egyptian-hieroglyphs/valid/"
test_folder = "/kaggle/input/egyptian-hieroglyphs/test/"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
            else:
                continue
    return images


train_images = load_images_from_folder(train_folder)

valid_images = load_images_from_folder(valid_folder)

test_images = load_images_from_folder(test_folder)

train_labels = train_data['class'].tolist()
valid_labels = valid_data['class'].tolist()
test_labels = test_data['class'].tolist()

train_images = np.array(train_images)
valid_images = np.array(valid_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
test_labels = np.array(test_labels)

from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
train_labels = encoder.fit_transform([train_labels])
valid_labels = encoder.fit_transform([valid_labels])
test_labels = encoder.fit_transform([test_labels])
train_labels = train_labels.flatten()
valid_labels = valid_labels.flatten()
test_labels = test_labels.flatten()
import gc
del(train_data)
del(valid_data)
del(test_data)

gc.collect()




train_images = train_images / 255.0
test_images = test_images / 255.0

# Определение размера изображений
image_size = 32

# Список гиперпараметров для тестирования
hyperparameters = [
    {'conv_filters': [75], 'pool_size': [(2, 2)], 'dense_units': [500]},
    {'conv_filters': [100], 'pool_size': [(2, 2)], 'dense_units': [600]},
    # Добавьте другие комбинации здесь
]


# Функция для создания и обучения модели
def create_and_train_model(hparams):
    model = Sequential()
    model.add(Conv2D(hparams['conv_filters'][0], kernel_size=(5, 5), activation='relu',
                     input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D(pool_size=hparams['pool_size'][0]))
    model.add(Dropout(0.2))

    model.add(Conv2D(hparams['conv_filters'][0], (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=hparams['pool_size'][0]))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(hparams['dense_units'][0], activation='relu'))
    model.add(Dropout(0.5))
    model.add(
        Dense(10, activation='softmax'))  # Измените количество выходных нейронов в соответствии с количеством категорий

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    return history.history['accuracy'][-1]  # Вернуть точность на последнем эпохе


# Тестирование моделей
results = []
for hparams in hyperparameters:
    accuracy = create_and_train_model(hparams)
    results.append((hparams, accuracy))

# Вывод результатов
print("Результаты:")
for i, result in enumerate(results):
    print(f"Комбинация {i + 1}: {result[0]}, Точность: {result[1]}")
