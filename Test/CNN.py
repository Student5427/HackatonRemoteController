import os
import json
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# Загрузка аннотации
with open('../DataSet/ESC_DATASET_v1.2/annotation/hr_bot_clear.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Путь к папке с аудиофайлами
audio_folder = '../DataSet/ESC_DATASET_v1.2/hr_bot_clear'

# Подготовка данных
audio_data = []
texts = []

for annotation in annotations:
    audio_filepath = os.path.join(audio_folder, annotation['audio_filepath'])
    audio, sr = librosa.load(audio_filepath, sr=None)

    # Извлечение MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    
    # Извлечение спектрограммы (мел-спектрограмма)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    
    # Преобразование в логарифмическую шкалу (опционально)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Паддинг до одинаковой длины
    max_length = 500  # Установите максимальную длину
    padded_mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), 'constant')
    padded_spectrogram = np.pad(mel_spectrogram_db, ((0, 0), (0, max_length - mel_spectrogram_db.shape[1])), 'constant')

    # Объединение MFCC и спектрограмм
    combined_features = np.concatenate((padded_mfcc, padded_spectrogram), axis=0)

    audio_data.append(combined_features)
    texts.append(annotation['text'])

# Преобразование в NumPy массив
audio_data = np.array(audio_data)
texts = np.array(texts)

# Кодирование текстов в метки
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(texts)

# Разделение данных
X_train, X_temp, y_train, y_temp = train_test_split(audio_data, y_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Изменение формы данных для CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))  # Добавление измерения канала
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# Создание модели CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc}')