import os
import numpy as np
import librosa
import tensorflow as tf
from keras import layers, models
import sounddevice as sd
import warnings
warnings.filterwarnings("ignore")

COMMANDS = ['stop', 'go', 'left']
DATA_DIR = 'data'  # Structure: data/stop/*.wav, data/go/*.wav, data/left/*.wav
SAMPLE_RATE = 16000
MFCC_FEATURES = 40
EPOCHS = 20
MODEL_PATH = 'command_model.h5'

def extract_features(file_path, max_pad_len=40):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

# --- DATA PREPARATION ---
def load_data():
    data, labels = [], []
    for idx, label in enumerate(COMMANDS):
        folder = os.path.join(DATA_DIR, label)
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                mfcc = extract_features(os.path.join(folder, file))
                data.append(mfcc)
                labels.append(idx)
    X = np.array(data)[..., np.newaxis]
    y = tf.keras.utils.to_categorical(labels, num_classes=len(COMMANDS))
    return X, y

# --- MODEL CREATION ---
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- AUDIO RECORDING ---
def record_audio(duration=1, fs=SAMPLE_RATE):
    print("Listening...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return np.squeeze(recording)

def predict_command(audio_data, model):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
    pad_width = 40 - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mfccs, verbose=0)
    return COMMANDS[np.argmax(prediction)]

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("Loading and training model...")
        X, y = load_data()
        model = build_model(input_shape=X.shape[1:], num_classes=len(COMMANDS))
        model.fit(X, y, epochs=EPOCHS, batch_size=32, validation_split=0.2)
        model.save(MODEL_PATH)
        print("Model trained and saved.")
    else:
        print("Loading saved model...")
        model = tf.keras.models.load_model(MODEL_PATH)

    while True:
        try:
            audio = record_audio()
            command = predict_command(audio, model)
            print(f"Predicted Command: {command.upper()}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
