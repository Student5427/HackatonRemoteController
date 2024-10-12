import librosa
import noisereduce as nr

from typing import Optional, Any, Tuple


def clean_audio(audio_file: str) -> Optional[Tuple[Any, int | float]]:
    """
    Функция очистки аудио от шумов

    :param audio_file: Путь к аудиофайлу
    :return: Массив значений амплитуды аудиосигнала и частота дискретизации
    """

    # Загрузка MP3 файла
    y, sr = librosa.load(audio_file, sr=16000)

    # Подавление шума
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    return y_denoised, sr
