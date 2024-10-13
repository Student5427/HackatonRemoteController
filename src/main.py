# import tkinter as tk
# from tkinter import filedialog
# import dearpygui.dearpygui as dpg

# import os
# import join
# from time import time
# from sklearn.metrics import f1_score as sklearn_f1_score

from clear_audio import clean_audio
from KerasModel import get_text
# from WhisperModel import get_text
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
    result = cls.predict("..\\add_data\\hr_bot_noise\\4e874bd6-76fe-11ee-85e3-c09bf4619c03.mp3")
    print(result)

""" Код вычислений метрик f1 и wer, а также среднего времени обработки аудио файла """

# def wer(ground_truth: str, prediction: str) -> float:
#     """Calculate the Word Error Rate (WER)."""
#     ground_truth_words = ground_truth.split()
#     prediction_words = prediction.split()
#
#     substitutions = sum(1 for i in range(len(ground_truth_words)) if i < len(prediction_words) and ground_truth_words[i] != prediction_words[i])
#     insertions = max(0, len(prediction_words) - len(ground_truth_words))
#     deletions = max(0, len(ground_truth_words) - len(prediction_words))
#
#     total_errors = substitutions + insertions + deletions
#     return total_errors / len(ground_truth_words) if ground_truth_words else 0.0
#
# if __name__ == "__main__":
#     cls = ModelPredict()
#
#     # Загрузка аннотаций
#     with open('../add_data/annotation/hr_bot_noise.json', 'r', encoding='utf-8') as f:
#         annotations = json.load(f)
#
#     true_labels = []
#     predicted_labels = []
#     true_texts = []
#     predicted_texts = []
#
#     time_avg = 0
#
#     # Перебор всех аудиофайлов в директории
#     for annotation in annotations:
#         audio_filepath = annotation["audio_filepath"]
#         audio_path = os.path.join('../add_data/hr_bot_noise', audio_filepath)
#
#         st = time()
#         # Получение предсказания модели
#         result = cls.predict(audio_path)
#         time_avg += time() - st
#
#         # Сохранение истинных и предсказанных значений
#         true_labels.append(annotation["label"])
#         predicted_labels.append(result["label"])
#         true_texts.append(annotation["text"])
#         predicted_texts.append(result["text"])
#
#     # Вычисление F1-метрики
#     f1_score_avg = sklearn_f1_score(true_labels, predicted_labels, average='weighted')
#
#     # Вычисление WER
#     wer_scores = [wer(true_texts[i], predicted_texts[i]) for i in range(len(true_texts))]
#     wer_avg = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0
#
#     print(f"Average F1 Score: {f1_score_avg:.4f}")
#     print(f"Average WER Score: {wer_avg:.4f}")
#     print(f'Время: {time_avg / len(annotations):.4f}')

""" Визуализация с tkinter """

# class App:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Audio Command Predictor")
#
#         self.label = tk.Label(root, text="Выберите аудиофайл:", font=("Arial", 14))
#         self.label.pack(pady=10)
#
#         self.select_button = tk.Button(root, text="Выбрать файл", command=self.load_file)
#         self.select_button.pack(pady=5)
#
#         self.result_frame = tk.Frame(root)
#         self.result_frame.pack(pady=20)
#
#         self.audio_name_label = tk.Label(self.result_frame, text="Название аудио: ", font=("Arial", 12))
#         self.audio_name_label.grid(row=0, column=0)
#
#         self.text_label = tk.Label(self.result_frame, text="Текст: ", font=("Arial", 12))
#         self.text_label.grid(row=1, column=0)
#
#         self.label_label = tk.Label(self.result_frame, text="Label: ", font=("Arial", 12))
#         self.label_label.grid(row=2, column=0)
#
#         self.attribute_label = tk.Label(self.result_frame, text="Attribute: ", font=("Arial", 12))
#         self.attribute_label.grid(row=3, column=0)
#
#     def load_file(self):
#         audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
#
#         if audio_path:
#             cls = ModelPredict()
#             result = cls.predict(audio_path)
#
#             # Обновление меток с результатами
#             self.audio_name_label.config(text=f"Название аудио: {audio_path.split('/')[-1]}")
#             self.text_label.config(text=f"Текст: {result['text']}")
#             self.label_label.config(text=f"Label: {result['label']}")
#             self.attribute_label.config(text=f"Attribute: {result['attribute']}")
#
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()

""" Визуализация с dearpygui """

# class App:
#     def __init__(self):
#         dpg.create_context()
#
#         with dpg.font_registry():
#             with dpg.font(r"..\fonts\caviar-dreams.ttf", 20) as font:
#                 dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
#                 dpg.bind_font(font)
#
#         # Заголовок окна
#         dpg.create_viewport(title="Audio Command Predictor", width=700, height=400)
#
#         # Компоненты интерфейса
#         with dpg.window(label="Audio Command Predictor", width=700, height=400):
#             self.select_button = dpg.add_button(label="Выбрать файл", callback=self.load_file)
#             self.result_frame = dpg.add_group(horizontal=False)
#
#             self.audio_name_label = dpg.add_text("Имя файла: ", parent=self.result_frame)
#             self.text_label = dpg.add_text("Текст: ", parent=self.result_frame)
#             self.label_label = dpg.add_text("Id команды: ", parent=self.result_frame)
#             self.attribute_label = dpg.add_text("Параметр: ", parent=self.result_frame)
#
#         dpg.create_viewport(title='Audio Command Predictor', width=700, height=400)
#         dpg.setup_dearpygui()
#         dpg.show_viewport()
#
#     def load_file(self):
#         audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
#
#         if audio_path:
#             cls = ModelPredict()
#             result = cls.predict(audio_path)
#
#             # Обновление меток с результатами
#             dpg.set_value(self.audio_name_label, f"Имя файла: {audio_path.split('/')[-1]}")
#             dpg.set_value(self.text_label, f"Текст: {result['text']}")
#             dpg.set_value(self.label_label, f"Id команды: {result['label']}")
#             dpg.set_value(self.attribute_label, f"Параметр: {result['attribute']}")
#
#
# if __name__ == "__main__":
#     app = App()
#     dpg.start_dearpygui()
