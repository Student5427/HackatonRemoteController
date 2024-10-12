import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import json
import os

# Загрузка модели
model = load_model('my_speech_recognition_model.keras')

# Загрузка аннотаций для создания LabelEncoder
with open('../DataSet/ESC_DATASET_v1.2/annotation/hr_bot_clear.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Подготовка списка текстов
texts = [annotation['text'] for annotation in annotations]
label_encoder = LabelEncoder()
label_encoder.fit(texts)  # Кодируем тексты

# Определение максимальной длины (используйте ту же, что была во время обучения)
max_length = 472  # Это значение должно совпадать с тем, что использовалось на этапе обучения.

def predict_audio(file_path, model, label_encoder, max_length):
    # Загрузка и предобработка аудиофайла
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Паддинг или обрезка MFCC до max_length
    if mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length]  # Обрезаем MFCC
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), 'constant')

    # Теперь надо изменить порядок размерностей для соответствия (1, max_length, 13)
    padded_mfcc = np.expand_dims(mfcc, axis=0)  # Это (1, 13, max_length)
    padded_mfcc = np.transpose(padded_mfcc, (0, 2, 1))  # Теперь это (1, max_length, 13)

    # Предсказание
    predicted_probs = model.predict(padded_mfcc)
    predicted_label = np.argmax(predicted_probs, axis=1)[0]

    # Получаем текст по метке
    predicted_text = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_text

# Предсказание текста из нового аудиофайла
audio_filepath = 'test.mp3'
predicted_text = predict_audio(audio_filepath, model, label_encoder, max_length)
print(f"Predicted text: {predicted_text}")
