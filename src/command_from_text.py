import re
import spacy

from typing import Tuple

import data.label2id as label2id

_nlp = spacy.load("ru_core_news_sm")

_num_dict = {
    "ноль": 0,
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "двадцать один": 21,
    "двадцать два": 22,
    "двадцать три": 23,
    "двадцать четыре": 24,
    "двадцать пять": 25,
    "двадцать шесть": 26,
    "двадцать семь": 27,
    "двадцать восемь": 28,
    "двадцать девять": 29,
    "тридцать": 30
}
_pattern_nums = rf"\b({'|'.join(re.escape(key) for key in _num_dict.keys())})\b"


def get_command(text: str) -> Tuple[int, int]:
    """
    Функция нахождения команды в тексте

    :param text: Исходный текст
    :return: Текст команды, id команды и ее параметр, иначе -1
    """

    text_lower = text.lower()

    # Лемматизация слов в тексте
    doc = _nlp(text_lower)
    words_text = ' '.join([token.lemma_ for token in doc if not token.is_stop])

    commands = label2id.get_labels()
    for command in commands:

        # Лемматизация слов в команде
        doc = _nlp(command)
        words_command = ' '.join([token.lemma_ for token in doc if not token.is_stop])

        # Проверка наличия команды в тексте
        if re.search(r'\( количество \)', words_command):
            pattern = words_command.replace("( количество )", _pattern_nums)
            find_command = re.search(pattern, words_text)
            if find_command:
                number = re.search(_pattern_nums, find_command.group(0))
                return label2id.label2id(command), _num_dict[number.group(0)]

        elif re.search(words_command, words_text):
            return label2id.label2id(command), -1

    return -1, -1
