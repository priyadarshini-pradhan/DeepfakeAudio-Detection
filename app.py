import os
import traceback
os.environ["NUMBA_DISABLE_JIT"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# Flask App Configuration
# -----------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "Models/deepfake_cnn.keras"
model = load_model(MODEL_PATH, compile=False)

print("Model Loaded Successfully!")
print("Output Shape:", model.output_shape)
print(type(model))

# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # -----------------------------
        # File check
        # -----------------------------
        if "audio" not in request.files:
            return "No audio file uploaded."

        file = request.files["audio"]

        if file.filename == "":
            return "No file selected."

        # -----------------------------
        # Save file
        # -----------------------------
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        print("FILE SAVED:", filepath)
        print("FILE EXISTS:", os.path.exists(filepath))

        # -----------------------------
        # Load audio (SAFE METHOD)
        # -----------------------------
        import soundfile as sf

        audio, sr = sf.read(filepath)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != 22050:
            raise Exception(f"Sampling rate is {sr}, expected 22050")

        print("AUDIO SHAPE:", audio.shape)
        print("SAMPLING RATE:", sr)

        # Trim to 5 sec
        max_length = sr * 5
        if len(audio) > max_length:
            audio = audio[:max_length]

        # -----------------------------
        # Waveform
        # -----------------------------
        plt.figure(figsize=(10, 3))
        plt.plot(audio)
        plt.title("Audio Waveform")

        waveform_filename = "waveform.png"
        waveform_path = os.path.join(app.static_folder, waveform_filename)

        plt.savefig(waveform_path, bbox_inches="tight")
        plt.close()

        print("Waveform saved:", os.path.exists(waveform_path))

        # -----------------------------
        # MFCC (TEMP FIX - DEBUG SAFE)
        # -----------------------------
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        print("MFCC SHAPE:", mfcc.shape)

        mfcc = np.mean(mfcc.T, axis=0)
        mfcc = mfcc.reshape(1, 40, 1)

        print("FINAL MFCC INPUT SHAPE:", mfcc.shape)

        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = model.predict(mfcc, verbose=0)

        print("PREDICTION:", prediction)

        label = np.argmax(prediction)
        confidence = round(float(np.max(prediction)) * 100, 2)

        result = "REAL AUDIO" if label == 0 else "FAKE AUDIO"

        # -----------------------------
        # Output file path
        # -----------------------------
        audio_file = url_for("static", filename="uploads/" + filename)

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            audio_file=audio_file,
            waveform_image=waveform_filename,
        )

    except Exception as e:
        print("ERROR OCCURRED:")
        traceback.print_exc()
        return f"<pre>{traceback.format_exc()}</pre>"


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)