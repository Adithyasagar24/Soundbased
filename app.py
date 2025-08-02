import os
import numpy as np
import librosa
import sounddevice as sd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# --- CONFIG ---
COMMANDS = ['stop', 'go', 'left']
SAMPLE_RATE = 16000
MODEL_PATH = 'command_model.h5'
DURATION = 1  # seconds
MFCC_FEATURES = 40
CHAR_SPEED = 20

# --- Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- Predict function ---
def predict_command(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
    pad_width = 40 - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mfccs, verbose=0)
    return COMMANDS[np.argmax(prediction)]

# --- Streamlit UI ---
st.title("Real-Time Voice Command with Playable Character")
st.markdown("Speak **go**, **stop**, or **left** and see the character move!")

if "x_pos" not in st.session_state:
    st.session_state.x_pos = 250
    st.session_state.command = "stop"

# Record button
if st.button("Record Command (1 sec)"):
    st.info("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    audio = np.squeeze(audio)
    command = predict_command(audio)
    st.success(f"Detected Command: **{command.upper()}**")
    st.session_state.command = command

    if command == "go":
        st.session_state.x_pos += CHAR_SPEED
    elif command == "left":
        st.session_state.x_pos -= CHAR_SPEED
    elif command == "stop":
        pass  # no movement

fig, ax = plt.subplots()
ax.set_xlim(0, 500)
ax.set_ylim(0, 100)
circle = plt.Circle((st.session_state.x_pos, 50), 15, color='skyblue')
ax.add_patch(circle)
ax.axis("off")
st.pyplot(fig)
