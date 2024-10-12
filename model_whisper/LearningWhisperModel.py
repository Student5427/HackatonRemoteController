import os
import json
import torch
import librosa
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Путь к файлу аннотации и папке с аудиофайлами
annotation_file = "../DataSet/ESC_DataSet_v1.2/annotation/hr_bot_clear.json"
audio_folder = "../DataSet/ESC_DataSet_v1.2/hr_bot_clear"

# Загрузка аннотаций
with open(annotation_file, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Извлечение путей к аудиофайлам и текстов
audio_paths = [os.path.join(audio_folder, annotation['audio_filepath']) for annotation in annotations]
texts = [annotation['text'] for annotation in annotations]

# Разделение данных на обучающую, валидационную и тестовую выборки
train_audio_paths, temp_audio_paths, train_texts, temp_texts = train_test_split(
    audio_paths, texts, test_size=0.2, random_state=42
)
val_audio_paths, test_audio_paths, val_texts, test_texts = train_test_split(
    temp_audio_paths, temp_texts, test_size=0.5, random_state=42
)

print(f"Train size: {len(train_audio_paths)}")
print(f"Validation size: {len(val_audio_paths)}")
print(f"Test size: {len(test_audio_paths)}")

# Загружаем процессор и модель Whisper
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

bos_token_id = processor.tokenizer.get_vocab().get("<|ru|>")  # Это ваш специальный токен для ru

# Функция для загрузки и обработки аудиофайлов
def load_audio_file(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    return audio.squeeze()

# Создание Dataset для обучения
def prepare_dataset(audio_paths, texts):
    dataset = Dataset.from_dict({"audio": audio_paths, "text": texts})
    return dataset

# Подготовка данных для модели
def process_data(batch):
    audio_inputs = []
    input_features = []
    labels = []

    for audio_path, text in zip(batch["audio"], batch["text"]):
        audio = None
        try:
            audio = load_audio_file(audio_path)
            audio_inputs.append(audio)
        except Exception:
            audio_inputs.append(None)

        if audio is not None:
            try:
                feature = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
                input_features.append(feature)
            except Exception:
                input_features.append(None)
        else:
            input_features.append(None)

        if text and isinstance(text, str) and text.strip():
            try:
                encoding = processor.tokenizer.encode_plus(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                label = encoding["input_ids"].squeeze(0)
                labels.append(label)
            except Exception:
                labels.append(None)
        else:
            labels.append(None)

    output = {
        "input_ids": labels,  # input_ids теперь добавлены
        "input_features": input_features,
    }
    return output

# Создание обучающей, валидационной и тестовой выборки
train_dataset = prepare_dataset(train_audio_paths, train_texts)
val_dataset = prepare_dataset(val_audio_paths, val_texts)
test_dataset = prepare_dataset(test_audio_paths, test_texts)

# Применение обработки к датасету
train_dataset = train_dataset.map(process_data, batched=True)
val_dataset = val_dataset.map(process_data, batched=True)
test_dataset = test_dataset.map(process_data, batched=True)

# Удаление ненужных столбцов (для оптимизации)
train_dataset = train_dataset.remove_columns(["audio"])
val_dataset = val_dataset.remove_columns(["audio"])
test_dataset = test_dataset.remove_columns(["audio"])

# Пользовательский DataCollator
class CustomDataCollator:
    def __call__(self, batch):
        input_ids = []
        input_features = []

        # Диагностика: проверяем каждый элемент батча
        for item in batch:
            if "input_ids" in item:
                input_ids.append(item["input_ids"])
            else:
                print("Warning: 'input_ids' key is missing in item:", item)
            input_features.append(item["input_features"])

        # Паддинг для input_ids
        padded_input_ids = pad_sequence(input_ids, batch_first=True)

        # Паддинг для input_features
        max_length = max(f.size(0) for f in input_features) if input_features else 0
        padded_input_features = torch.stack(
            [torch.nn.functional.pad(f, (0, 0, 0, max_length - f.size(0))) for f in input_features]
        )

        print(padded_input_ids)  # Оставлен только этот вывод

        return {
            "input_ids": padded_input_ids,
            "input_features": padded_input_features,
        }

data_collator = CustomDataCollator()

# Конфигурация аргументов обучения
training_args = TrainingArguments(
    output_dir="./whisper-fine-tuned",
    per_device_train_batch_size=2,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    logging_dir='./logs',
)

# Создание тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Запуск обучения
try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")

# Сохранение модели
#trainer.save_model("whisper-fine-tuned")
# Сохранение модели
model.save_pretrained("whisper-fine-tuned")

# Сохранение процессора
processor.save_pretrained("whisper-fine-tuned")

# Оценка на тестовой выборке
test_results = trainer.evaluate(test_dataset)
print(f"Test results: {test_results}")
