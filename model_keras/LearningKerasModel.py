import os
import json
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import pickle  # Импортируем библиотеку для сохранения объектов
import time

# Загрузка аннотаций по чистым файлам
with open('../DataSet/ESC_DATASET_v1.2/annotation/hr_bot_clear.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Путь к папке с аудиофайлами
audio_folder = '../DataSet/ESC_DATASET_v1.2/hr_bot_clear'

# Подготовка данных
audio_data = []
texts = []

# Получение максимальной длины MFCC для предоставления одинакового размера
max_length = 0

# Сначала определим максимальную длину
for annotation in annotations:
    audio_filepath = os.path.join(audio_folder, annotation['audio_filepath'])
    audio, sr = librosa.load(audio_filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    audio_data.append(mfcc)
    texts.append(annotation['text'])
    max_length = max(max_length, mfcc.shape[1])  # Обновляем максимальную длину

# Теперь добавим паддинг
padded_audio_data = []
for mfcc in audio_data:
    # Паддинг до max_length
    padded_mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), 'constant')
    padded_audio_data.append(padded_mfcc)

# Преобразование в фиксированный массив NumPy
padded_audio_data = np.array(padded_audio_data)
texts = np.array(texts)

# Кодирование текстов в метки
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(texts)

# Сохранение LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Сохранение max_length
with open('max_length.txt', 'w') as f:
    f.write(str(max_length))

# Разделение данных: 80% на обучение, 10% на валидацию и 10% на тест
X_train, X_temp, y_train, y_temp = train_test_split(padded_audio_data, y_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 10% валидация, 10% тест

def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(320))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Параметры модели
num_classes = len(label_encoder.classes_)  # Количество уникальных текстовых меток
input_shape = (padded_audio_data.shape[1], padded_audio_data.shape[2])  # (13, max_length)

# Создание модели
with tf.device('/GPU:0'):
    model = create_model(input_shape, num_classes)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Вывод структуры модели
model.summary()

# Обучение модели
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Сохранение модели
model.save('my_speech_recognition_model.keras')

# Оценка модели
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")
