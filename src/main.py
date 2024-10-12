from clear_audio import clean_audio
from WhisperModel import get_text
from command_from_text import get_command


class ModelPredict:

    @classmethod
    def predict(cls, audio_path: str) -> dict[str: str | int]:

        # Очистка звука
        amplitude, sf = clean_audio(audio_path)

        # Преобразование из аудио в текст
        text = get_text(amplitude, sf)

        # Получение из текста команды
        id_command, attribute = get_command(text)

        return {
            "text": text,
            "label": id_command,
            "attribute": attribute
        }


# Основной процесс
if __name__ == "__main__":
    cls = ModelPredict()
    result = cls.predict("..\\add_data\\hr_bot_noise\\2d8bef86-76fe-11ee-96b9-c09bf4619c03.mp3")
    print(result)
