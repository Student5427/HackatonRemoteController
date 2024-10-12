import os
import json
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt

# Загрузка аннотаций по чистым файлам
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
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Извлечение спектрограммы (мел-спектрограмма)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    
    # Преобразование в логарифмическую шкалу
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

# Создание модели
def build_model(hp):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(padded_mfcc.shape[0], padded_mfcc.shape[1])))

    model.add(layers.LSTM(hp.Int('units', min_value=64, max_value=512, step=64), return_sequences=True))
    model.add(layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))

    model.add(layers.LSTM(hp.Int('units2', min_value=64, max_value=512, step=64)))
    model.add(layers.Dense(len(label_encoder.classes_), activation=hp.Choice('activation', ['softmax', 'sigmoid'])))

    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                  loss=hp.Choice('loss', ['sparse_categorical_crossentropy', 'categorical_crossentropy']),
                  metrics=['accuracy'])

    return model

# Создание экземпляра Keras Tuner и запуск подбора параметров
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='my_dir',
    project_name='speech_recognition_tuning'
)

# Остановка по раннему завершению
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Поиск лучших гиперпараметров
tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[stop_early])

# Получение и печать лучших гиперпараметров и метрик
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
best_results = tuner.get_best_models(num_models=1)[0]

print(f"""
    Найденные лучшие параметры:
    - Количество единиц в LSTM (первый слой): {best_hyperparameters.get('units')}
    - Количество единиц в LSTM (второй слой): {best_hyperparameters.get('units2')}
    - Дропаут: {best_hyperparameters.get('dropout')}
    - Оптимизатор: {best_hyperparameters.get('optimizer')}
    - Активация: {best_hyperparameters.get('activation')}
    - Функция потерь: {best_hyperparameters.get('loss')}

    Лучшие метрики:
    - Тестовая точность: {best_results.metrics["val_accuracy"]} 
    - Тестовая потеря: {best_results.metrics["val_loss"]}
""")

# Создание и обучение модели с лучшими гиперпараметрами
model = tuner.hypermodel.build(best_hyperparameters)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))