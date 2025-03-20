# Speech-to-Text Using Deep Learning (RNN + LSTM + Wav2Vec2)
# Project Overview
This project is a deep learning-based speech-to-text system that converts spoken words into text. It utilizes RNN and LSTM models along with the Wav2Vec2 transformer model for high-accuracy transcription. The dataset used for training is sourced from Kaggle, containing a large collection of audio files.

# Step-by-Step Explanation
1. Data Collection and Preprocessing
The dataset, which consists of audio recordings, is downloaded from Kaggle.
The .wav audio files are stored in a structured directory.
The librosa library is used to load and process audio files.
Each audio file is converted into Mel-Frequency Cepstral Coefficients (MFCCs), which serve as input features for the LSTM model.
2. Model Development
The project uses two key models:
Wav2Vec2: A transformer-based speech recognition model for feature extraction.
LSTM Model: A sequential deep learning model trained on MFCC features to recognize speech patterns.
The LSTM model consists of:
Two LSTM layers for sequential feature learning.
Dropout layers to prevent overfitting.
A dense output layer with softmax activation for classification.
3. Model Training
The processed audio features are fed into the LSTM model.
The model is compiled using categorical cross-entropy loss and the Adam optimizer.
Training is performed over multiple epochs with a batch size of 16.
4. Model Saving
After training, the LSTM model is saved as "lstm_model.h5" for future use.
The Wav2Vec2 processor is also saved using pickle.
5. Testing and Deployment
A separate Python script (main.py) is created to accept user input.
The script loads the trained LSTM model and Wav2Vec2 processor.
The user provides an audio file (.wav), which is converted into text using the Wav2Vec2 model.
The transcribed text is displayed as output.
6. Running the Project
# Clone the repository.
Install dependencies using pip install -r requirements.txt.
Run python main.py and provide an audio file for transcription.
Technologies Used
Python
TensorFlow/Keras
Librosa
Transformers (Hugging Face)
Pickle (for model storage)
