
# HackatonRemoteController

HackatonRemoteController — это проект, предназначенный для определения голосовых команд. Он позволяет пользователям взаимодействовать с устройствами с помощью голосовых команд, что упрощает управление и делает его более интуитивным.

## Структура проекта

```markdown
HackatonRemoteController
|
|- data
|    |- label2id.py              # Набор команд и их ID
|- src
|    |
|    |- main.py                  # Запуск файла
|    |- clear_audio.py           # Очищение аудио от шумов
|    |- KerasModel.py            # Распознавание текста из аудио
|    |- command_from_text.py     # Преобразование из текста в команду
|    |- WhisperModel.py          # Альтернативная модел распознавания текста из аудио
|- model_keras
|    |
|    |- LeaningKerasModel.py     # Файл обучения модели распознавания текста из аудио
|    |- *.*                      # Файлы работы модели
|- model_whisper
|    |
|    |- LeaningWhisperModel.py     # Файл обучения модели распознавания текста из аудио
|
|- get_submissions.py            # Скрипт для получения сабмитов
|- pyproject.toml                # Установка окружения
|- requirements.txt              # Зависимости
|- README.md                     # Документация проекта
```

## Установка окружения

Для установки окружения проекта выполните следующие шаги:

### 1. Установите Poetry

Если у вас еще не установлен Poetry, вы можете установить его, следуя [официальной документации](https://python-poetry.org/docs/#installation).

### 2. Клонируйте репозиторий

Сначала клонируйте репозиторий проекта:

   ```bash
   git clone https://github.com/Student5427/HackatonRemoteController
   cd hackatonremotecontroller
   ```

### 3. Установите зависимости
Установите все необходимые зависимости, используя Poetry:
   ```bash
   poetry install
   ```

## Использование

Для запуска проекта выполните следующую команду:
```bash
poetry run python main.py
```

## Возможности

- Определение голосовых команд.
- Очищение аудио от шумов для повышения точности распознавания.
- Преобразование текста в команды для дальнейшей обработки.

## Участники

- Копосов Андрей
- Хайдарова Алиса
- Потемкина Татьяна
- Скуртенко Шади
- Самсонов Евгений
