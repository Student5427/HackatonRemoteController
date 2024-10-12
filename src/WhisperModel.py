import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from typing import Any


def get_text(audio: Any, sf: int | float) -> str:
    """
    Функция преобразования аудиофайла в текст

    :param audio: Массив значений амплитуды аудиосигнала
    :param sf: Частота дискретизации
    :return: Обнаруженный текст
    """

    # Загрузка модели и процессора из сохраненной директории
    model = WhisperForConditionalGeneration.from_pretrained("../whisper-fine-tuned")
    processor = WhisperProcessor.from_pretrained("../whisper-fine-tuned")

    # Подготовка аудиофайла для модели
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    # Вызов специального токена для русского языка, если он доступен
    bos_token_id = processor.tokenizer.get_vocab()["<|ru|>"]

    # Получение предсказаний
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], forced_bos_token_id=bos_token_id)

    # Декодирование предсказаний в текст
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # Вывод результатов
    return transcription[0]
