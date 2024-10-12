from typing import Any

import librosa
import numpy as np
import tensorflow as tf
import pickle

# ### Загрузка модели и необходимых переменных ###

# Загрузка модели
loaded_model = tf.keras.models.load_model('..\\model_keras\\my_speech_recognition_model.keras')

# Загрузка LabelEncoder
with open('../model_keras/label_encoder.pkl', 'rb') as f:
    loaded_label_encoder = pickle.load(f)

# Загрузка max_length
with open('../model_keras/max_length.txt', 'r') as f:
    max_length = int(f.read().strip())


# ### Функция предсказания ###
def predict_audio(audio: Any, sr: int | float, model: Any, label_encoder: Any) -> str:
    """
    Функция преобразования аудиофайла в текст

    :param audio: Массив значений амплитуды аудиосигнала
    :param sr: Частота дискретизации
    :param model: Модель преобразования
    :param label_encoder: Декодер метрик в текст
    :return: Обнаруженный текст
    """
    # Предобработка аудиофайла
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Паддинг MFCC до максимальной длины, использованной в обучении
    padded_mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), 'constant')
    
    # Создание 3D массива для совместимости с моделью
    padded_mfcc = np.expand_dims(padded_mfcc, axis=0)

    # Предсказание
    predicted_probs = model.predict(padded_mfcc)
    predicted_label = np.argmax(predicted_probs, axis=1)[0]

    # Получаем текст по метке
    predicted_text = label_encoder.inverse_transform([predicted_label])[0]
    
    return predicted_text


# ### Запуск предсказания ###
def get_text(audio, sr):
    predicted_text = predict_audio(audio, sr, loaded_model, loaded_label_encoder)
    return predicted_text
