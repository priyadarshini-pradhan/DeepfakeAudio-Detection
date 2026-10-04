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
import soundfile as sf

from tensorflow.keras.models import load_model


# ==========================================
# Flask App Configuration
# ==========================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ==========================================
# V2 Parameters
# ==========================================

N_MFCC = 40
MAX_SECONDS = 5
MAX_FRAMES = 400


# ==========================================
# Load V2 Model
# ==========================================

MODEL_PATH = "Models/deepfake_cnn_v2.keras"

model = load_model(
    MODEL_PATH,
    compile=False
)

print("================================")
print("V2 MODEL LOADED SUCCESSFULLY")
print("================================")

print("Model Path:", MODEL_PATH)
print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)


# ==========================================
# Home
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# Prediction Route
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ==================================
        # Check uploaded file
        # ==================================

        if "audio" not in request.files:

            return "No audio file uploaded."

        file = request.files["audio"]

        if file.filename == "":

            return "No file selected."


        # ==================================
        # Save uploaded file
        # ==================================

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        print("\n================================")
        print("NEW AUDIO FILE")
        print("================================")

        print(
            "FILE SAVED:",
            filepath
        )


        # ==================================
        # Load audio
        # ==================================

        audio, sr = sf.read(
            filepath
        )


        # Stereo → Mono

        if len(audio.shape) > 1:

            audio = np.mean(
                audio,
                axis=1
            )


        audio = audio.astype(
            np.float32
        )


        print(
            "Original Audio Shape:",
            audio.shape
        )

        print(
            "Sampling Rate:",
            sr
        )


        # ==================================
        # Keep original sample rate
        # ==================================
        #
        # IMPORTANT:
        # V2 training/evaluation did NOT
        # resample the audio.
        #

        if sr != 16000:

            print(
                "WARNING: Unexpected "
                f"sampling rate: {sr}"
            )


        # ==================================
        # Trim to first 5 seconds
        # ==================================

        max_length = (
            sr * MAX_SECONDS
        )

        if len(audio) > max_length:

            audio = audio[
                :max_length
            ]


        print(
            "Trimmed Audio Shape:",
            audio.shape
        )


        # ==================================
        # Waveform
        # ==================================

        plt.figure(
            figsize=(10, 3)
        )

        plt.plot(audio)

        plt.title(
            "Audio Waveform"
        )

        plt.xlabel(
            "Samples"
        )

        plt.ylabel(
            "Amplitude"
        )

        waveform_filename = (
            "waveform.png"
        )

        waveform_path = os.path.join(
            app.static_folder,
            waveform_filename
        )

        plt.savefig(
            waveform_path,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Waveform saved:",
            waveform_path
        )


        # ==================================
        # V2 MFCC Extraction
        # ==================================

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        print(
            "Original MFCC Shape:",
            mfcc.shape
        )


        # ==================================
        # Pad / Crop to 400 frames
        # ==================================

        if mfcc.shape[1] < MAX_FRAMES:

            pad_width = (
                MAX_FRAMES -
                mfcc.shape[1]
            )

            mfcc = np.pad(
                mfcc,
                (
                    (0, 0),
                    (0, pad_width)
                ),
                mode="constant"
            )

        else:

            mfcc = mfcc[
                :,
                :MAX_FRAMES
            ]


        print(
            "MFCC Shape after "
            "padding/cropping:",
            mfcc.shape
        )


        # ==================================
        # Add channel dimension
        # ==================================

        mfcc = mfcc[
            ...,
            np.newaxis
        ]


        # ==================================
        # Add batch dimension
        # ==================================

        mfcc = mfcc[
            np.newaxis,
            ...
        ]


        print(
            "FINAL V2 INPUT SHAPE:",
            mfcc.shape
        )


        # Expected:
        #
        # (1, 40, 400, 1)


        # ==================================
        # Prediction
        # ==================================

        prediction = model.predict(
            mfcc,
            verbose=0
        )


        print(
            "PREDICTION:",
            prediction
        )


        # ==================================
        # Class
        # ==================================

        label = np.argmax(
            prediction
        )


        # Class 0 = Bonafide
        # Class 1 = Spoof

        bonafide_score = float(
            prediction[0][0]
        )

        spoof_score = float(
            prediction[0][1]
        )


        # ==================================
        # Result
        # ==================================

        if label == 0:

            result = "REAL AUDIO"

            confidence = round(
                bonafide_score * 100,
                2
            )

        else:

            result = "FAKE AUDIO"

            confidence = round(
                spoof_score * 100,
                2
            )


        print(
            "RESULT:",
            result
        )

        print(
            "Confidence:",
            confidence,
            "%"
        )


        # ==================================
        # Audio URL
        # ==================================

        audio_file = url_for(
            "static",
            filename=(
                "uploads/" + filename
            )
        )


        # ==================================
        # Return Result
        # ==================================

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            audio_file=audio_file,
            waveform_image=waveform_filename
        )


    except Exception as e:

        print(
            "ERROR OCCURRED:"
        )

        traceback.print_exc()

        return (
            f"<pre>"
            f"{traceback.format_exc()}"
            f"</pre>"
        )


# ==========================================
# Run Flask App
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )