import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Укажите путь к вашему аудиофайлу
audio_file_path = "test5.wav"

# Загрузка модели и процессора из сохраненной директории
model = WhisperForConditionalGeneration.from_pretrained("whisper-fine-tuned")
processor = WhisperProcessor.from_pretrained("whisper-fine-tuned")

# Функция для загрузки и обработки аудиофайлов
def load_audio_file(filepath):
    audio, sr = librosa.load(filepath, sr=16000)  # Загружаем аудио
    return audio

# Загрузка и обработка аудиофайла
audio = load_audio_file(audio_file_path)

# Подготовка аудиофайла для модели
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# Вызов специального токена для русского языка, если он доступен
# Убедитесь, что этот токен существует
bos_token_id = processor.tokenizer.get_vocab()["<|ru|>"]  # Замените "ru" на нужный специальный токен для русского языка

# Получение предсказаний
with torch.no_grad():
    predicted_ids = model.generate(inputs["input_features"], forced_bos_token_id=bos_token_id)

# Декодирование предсказаний в текст
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# Вывод результатов
print("Transcription:", transcription)