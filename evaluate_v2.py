import os
import random
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# =================================
# Paths
# =================================

MODEL_PATH = "Models/deepfake_cnn_v2.keras"

PROTOCOL_PATH = (
    "DataSet/LA/LA/ASVspoof2019_LA_cm_protocols/"
    "ASVspoof2019.LA.cm.eval.trl.txt"
)

AUDIO_FOLDER = (
    "DataSet/LA/LA/ASVspoof2019_LA_eval/flac"
)

RESULT_FOLDER = "Results/V2"

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# =================================
# Parameters
# =================================

N_MFCC = 40
MAX_SECONDS = 5
MAX_FRAMES = 400


# =================================
# Load Model
# =================================

print("Loading V2 model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("V2 Model Loaded Successfully!")

print(
    "Input Shape:",
    model.input_shape
)

print(
    "Output Shape:",
    model.output_shape
)


# =================================
# Read Evaluation Protocol
# =================================

print("\nReading evaluation protocol...")

with open(
    PROTOCOL_PATH,
    "r"
) as f:

    lines = f.readlines()

print(
    "Total protocol entries:",
    len(lines)
)


# =================================
# Shuffle
# =================================

random.seed(42)
random.shuffle(lines)


# =================================
# Storage
# =================================

y_true = []
y_pred = []
y_scores = []
file_names = []

processed = 0
skipped = 0


# =================================
# Process Audio
# =================================

for line in lines:

    parts = line.strip().split()

    if len(parts) < 5:

        skipped += 1
        continue

    file_name = parts[1] + ".flac"
    label = parts[-1]

    audio_path = os.path.join(
        AUDIO_FOLDER,
        file_name
    )

    if not os.path.exists(audio_path):

        skipped += 1
        continue

    # Bonafide = 0
    # Spoof = 1

    true_label = (
        0 if label == "bonafide" else 1
    )

    try:

        # =================================
        # Read Audio
        # =================================

        audio, sr = sf.read(
            audio_path
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


        # =================================
        # Sampling Rate
        # =================================

        # Training did not resample.
        # Keep original sampling rate.

        if sr != 16000:

            print(
                f"Unexpected sample rate: "
                f"{sr} | {file_name}"
            )


        # =================================
        # First 5 Seconds
        # =================================

        max_length = (
            sr * MAX_SECONDS
        )

        if len(audio) > max_length:

            audio = audio[:max_length]


        # =================================
        # MFCC
        # =================================

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )


        # =================================
        # Preserve Time Dimension
        # =================================

        if mfcc.shape[1] < MAX_FRAMES:

            pad_width = (
                MAX_FRAMES - mfcc.shape[1]
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


        # =================================
        # Add Channel Dimension
        # =================================

        mfcc = mfcc[
            ...,
            np.newaxis
        ]

        # Shape:
        # (40, 400, 1)

        mfcc = mfcc[
            np.newaxis,
            ...
        ]

        # Final shape:
        # (1, 40, 400, 1)


        # =================================
        # Prediction
        # =================================

        prediction = model.predict(
            mfcc,
            verbose=0
        )

        predicted_label = np.argmax(
            prediction
        )

        spoof_score = float(
            prediction[0][1]
        )


        # =================================
        # Store
        # =================================

        y_true.append(
            true_label
        )

        y_pred.append(
            predicted_label
        )

        y_scores.append(
            spoof_score
        )

        file_names.append(
            file_name
        )

        processed += 1


        # =================================
        # Progress
        # =================================

        if processed % 500 == 0:

            print(
                f"Processed: "
                f"{processed} / {len(lines)}"
            )


    except Exception as e:

        skipped += 1

        print(
            "Skipped:",
            file_name,
            "|",
            str(e)
        )


# =================================
# Evaluation Completed
# =================================

print("\n================================")
print("V2 EVALUATION COMPLETED")
print("================================")

print(
    "Successfully processed:",
    len(y_true)
)

print(
    "Skipped:",
    skipped
)


# =================================
# Metrics
# =================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# =================================
# ROC-AUC
# =================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_scores
)

fnr = 1 - tpr

eer_index = np.nanargmin(
    np.abs(fnr - fpr)
)

eer = fpr[eer_index]

auc = roc_auc_score(
    y_true,
    y_scores
)


# =================================
# Print Results
# =================================

print(
    f"\nROC-AUC  : {auc:.4f}"
)

print(
    f"EER      : {eer * 100:.2f}%"
)

print(
    "\n===== V2 MODEL PERFORMANCE ====="
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# =================================
# Classification Report
# =================================

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Bonafide",
        "Spoof"
    ],
    zero_division=0
)

print(
    "\n===== CLASSIFICATION REPORT ====="
)

print(report)


# =================================
# Save Report
# =================================

report_path = os.path.join(
    RESULT_FOLDER,
    "evaluation_report_v2.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "ASVspoof 2019 LA - V2 Evaluation\n\n"
    )

    f.write(
        f"Processed samples: "
        f"{len(y_true)}\n"
    )

    f.write(
        f"Skipped samples: "
        f"{skipped}\n\n"
    )

    f.write(
        f"ROC-AUC  : {auc:.4f}\n"
    )

    f.write(
        f"EER      : {eer * 100:.2f}%\n"
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score : {f1:.4f}\n\n"
    )

    f.write(report)


# =================================
# Confusion Matrix
# =================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print(
    "Confusion Matrix:"
)

print(cm)


# =================================
# Save Prediction Details
# =================================

prediction_path = os.path.join(
    RESULT_FOLDER,
    "prediction_details_v2.csv"
)

with open(
    prediction_path,
    "w"
) as f:

    f.write(
        "file_name,true_label,"
        "predicted_label,spoof_score\n"
    )

    for name, true, pred, score in zip(
        file_names,
        y_true,
        y_pred,
        y_scores
    ):

        f.write(
            f"{name},{true},"
            f"{pred},{score:.6f}\n"
        )


print(
    "Prediction details saved."
)


# =================================
# Save Confusion Matrix
# =================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Bonafide",
        "Spoof"
    ]
)

disp.plot()

plt.title(
    "ASVspoof 2019 LA - V2 Evaluation"
)

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "confusion_matrix_v2.png"
    ),
    bbox_inches="tight"
)

plt.close()


# =================================
# Final
# =================================

print(
    "\nV2 results saved inside:"
)

print(
    RESULT_FOLDER
)