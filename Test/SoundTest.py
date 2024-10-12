import librosa
import noisereduce as nr
import soundfile as sf

# Загружаем аудиофайл
file_path = 'test.mp3'
audio, sr = librosa.load(file_path, sr=None)

# Применяем шумоподавление
reduced_noise = nr.reduce_noise(y=audio, sr=sr)

# Сохраняем очищенный аудиофайл
output_path = 'cleaned_audio_file.wav'
sf.write(output_path, reduced_noise, sr)

print("Звук очищен от шумов и сохранен как", output_path)