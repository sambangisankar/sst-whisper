import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC
import pickle

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
wav2vec_model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h", from_pt=True)

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def audio_to_mfcc(file_path, target_sr=16000, max_pad_len=100):
    y, sr = librosa.load(file_path, sr=target_sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc.T

def find_audio_files(folder):
    audio_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    return audio_files

audio_folder = "C:/Users/hp/Desktop/dlproject/newd"
audio_files = find_audio_files(audio_folder)

X, Y = [], []

for i, file_path in enumerate(audio_files):
    mfcc = audio_to_mfcc(file_path)
    X.append(mfcc)
    Y.append(i)

X = np.array(X)
Y = np.array(Y)

input_shape = (X.shape[1], X.shape[2])
num_classes = len(set(Y))
lstm_model = build_lstm_model(input_shape, num_classes)

lstm_model.fit(X, Y, epochs=10, batch_size=16)
lstm_model.save("lstm_model.h5")

with open("wav2vec_processor.pkl", "wb") as f:
    pickle.dump(processor, f)

print("Models saved successfully!")
