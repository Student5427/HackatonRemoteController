import data.annotation.label2id as label2id
from clear_audio import clean_audio
from command_from_text import find_command_in_text

from decorates import time_memory


@time_memory
def main():
    audio_filename = "..\\data\\hr_bot_noise\\2d8bef86-76fe-11ee-96b9-c09bf4619c03.mp3"

    # Очистка звука
    cleaned_audio = clean_audio(audio_filename)

    # Получение из текста набор команд
    commands = label2id._label2id
    command = find_command_in_text("привет локоматив нужно осадить на 16 вагонов сделай пожалуйста", commands)
    if isinstance(command, tuple):
        print(f'Команда: {command[0]}\nАргумент: {command[1]}')
    else:
        print(f'Команда: {command}')


# Основной процесс
if __name__ == "__main__":
    main()
