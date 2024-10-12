# import tkinter as tk
# from tkinter import filedialog
# import dearpygui.dearpygui as dpg

from clear_audio import clean_audio
from KerasModel import get_text
# from WhisperModel import get_text
from command_from_text import get_command

from time import time


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
