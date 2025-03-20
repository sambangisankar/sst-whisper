import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC
import pickle

lstm_model = load_model("lstm_model.h5")

with open("wav2vec_processor.pkl", "rb") as f:
    processor = pickle.load(f)

wav2vec_model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h", from_pt=True)

def audio_to_mfcc(file_path, target_sr=16000, max_pad_len=100):
    y, sr = librosa.load(file_path, sr=target_sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc.T

def transcribe_audio(file_path):
    speech, sr = librosa.load(file_path, sr=16000)

    if len(speech) == 0:
        return "[ERROR] Empty or invalid audio file."

    input_values = processor(speech, return_tensors="tf", sampling_rate=16000).input_values
    logits = wav2vec_model(input_values).logits
    predicted_ids = tf.argmax(logits, axis=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

user_audio = input("Enter the path of the audio file: ")

if os.path.exists(user_audio) and user_audio.endswith(".wav"):
    print(f"Processing: {os.path.basename(user_audio)}")
    
    text = transcribe_audio(user_audio)
    print(f"Transcription: {text}\n")
    
    mfcc = audio_to_mfcc(user_audio)
    mfcc = np.expand_dims(mfcc, axis=0)
    
    prediction = lstm_model.predict(mfcc)
    predicted_class = np.argmax(prediction)

    print(f"Predicted Class: {predicted_class}")

else:
    print("Invalid file path. Please enter a valid .wav file path.")
