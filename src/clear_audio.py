import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import soundfile as sf

import os
from decorates import time_memory
from typing import Optional, Any


@time_memory
def clean_audio(audio_file: str) -> Optional[tuple[Any, int | float]]:
    if not os.path.exists(audio_file):
        print('Файл не найден')
        return None

    # Загрузка MP3 файла
    y, sr = librosa.load(audio_file, sr=None)

    # Подавление шума
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    # Визуализация
    # plt.figure(figsize=(12, 8))
    #
    # # Оригинальный сигнал
    # plt.subplot(2, 1, 1)
    # librosa.display.waveshow(y, sr=sr, alpha=0.5)
    # plt.title('Оригинальный сигнал')
    # plt.xlabel('Время (с)')
    # plt.ylabel('Амплитуда')
    #
    # # Очищенный сигнал
    # plt.subplot(2, 1, 2)
    # librosa.display.waveshow(y_denoised, sr=sr, color='orange', alpha=0.5)
    # plt.title('Очищенный сигнал')
    # plt.xlabel('Время (с)')
    # plt.ylabel('Амплитуда')
    #
    # plt.tight_layout()
    # plt.show()

    return y_denoised, sr
