import re
import spacy

from typing import Optional
from decorates import time_memory

_nlp = spacy.load("ru_core_news_sm")


@time_memory
def find_command_in_text(text: str, commands: dict[str: int]) -> Optional[str | tuple[str, int]]:

    text_lower = text.lower()

    doc = _nlp(text_lower)
    words_text = ' '.join([token.lemma_ for token in doc if not token.is_stop])

    for command, _ in commands.items():
        doc = _nlp(command)
        words_command = ' '.join([token.lemma_ for token in doc if not token.is_stop])

        if re.search(r'\( количество \)', words_command):
            pattern = words_command.replace("( количество )", r"(\d+)")
            find_command = re.search(pattern, words_text)[0]
            if find_command:
                number = re.search(r'(\d+)', find_command)[0]
                return command, int(number)
        else:
            pattern = words_command

        if re.search(pattern, words_text):
            return command

    return None
