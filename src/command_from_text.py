import re
import spacy

from typing import Union, Tuple
from decorates import time_memory

_nlp = spacy.load("ru_core_news_sm")


@time_memory
def find_command_in_text(text: str, commands: list) -> Union[Tuple[int, int] | Tuple[str, int]]:
    """
    Функция нахождения команды в тексте

    :param text: Исходный текст
    :param commands: Набор команд
    :return: Команду и её параметр, иначе -1
    """

    text_lower = text.lower()

    # Лемматизация слов в тексте
    doc = _nlp(text_lower)
    words_text = ' '.join([token.lemma_ for token in doc if not token.is_stop])

    for command in commands:

        # Лемматизация слов в команде
        doc = _nlp(command)
        words_command = ' '.join([token.lemma_ for token in doc if not token.is_stop])

        # Проверка наличия команды в тексте
        if re.search(r'\( количество \)', words_command):
            pattern = words_command.replace("( количество )", r"(\d+)")
            find_command = re.search(pattern, words_text)[0]
            if find_command:
                number = re.search(r'(\d+)', find_command)[0]
                return command, int(number)

        elif re.search(words_command, words_text):
            return command, -1

    return -1, -1
