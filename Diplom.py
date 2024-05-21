import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка и подготовка данных
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
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
