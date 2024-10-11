import pyaudio
import pyttsx3

import data.annotation.label2id as label2id
from clear_audio import clean_audio

# Настройки аудио
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024


# Функция для выполнения команды
def execute_command(command):

    id_command = label2id.label2id(command)
    response = label2id.id2label(id_command) if id_command else "Команда не распознана."

    # Голосовой ответ
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()


def main():
    audio_filename = "..\\data\\hr_bot_noise\\2d8bef86-76fe-11ee-96b9-c09bf4619c03.mp3"

    cleaned_audio = clean_audio(audio_filename)  # Очистка звука (можно доработать для сохранения)


# Основной процесс
if __name__ == "__main__":
    main()
