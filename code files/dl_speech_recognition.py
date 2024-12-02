import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('Hello')

def extract_mfcc(audio_path, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# Example usage
mfcc = extract_mfcc("CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav")
print("MFCC Shape:", mfcc.shape)

def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# Example CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 100, 1)),  # Example input shape
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')  # 6 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

