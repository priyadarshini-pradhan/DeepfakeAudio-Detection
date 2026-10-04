import os
import numpy as np
import librosa
import soundfile as sf


# ============================================
# Paths
# ============================================
PROTOCOL_PATH = (
    "DataSet/LA/LA/ASVspoof2019_LA_cm_protocols/"
    "ASVspoof2019.LA.cm.train.trn.txt"
)

AUDIO_FOLDER = (
    "DataSet/LA/LA/"
    "ASVspoof2019_LA_train/flac"
)

OUTPUT_X = "Features/X_v2.npy"
OUTPUT_Y = "Features/y_v2.npy"


# ============================================
# Parameters
# ============================================

N_MFCC = 40
MAX_SECONDS = 5

# Fixed number of MFCC time frames
MAX_FRAMES = 400


# ============================================
# Create output folder
# ============================================

os.makedirs("Features", exist_ok=True)


# ============================================
# Read Protocol
# ============================================

print("Reading training protocol...")

with open(PROTOCOL_PATH, "r") as f:
    lines = f.readlines()

print("Total protocol entries:", len(lines))


# ============================================
# Storage
# ============================================

X = []
y = []

processed = 0
skipped = 0


# ============================================
# Process Audio
# ============================================

for i, line in enumerate(lines):

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

    try:

        # ------------------------------------
        # Read audio
        # ------------------------------------

        audio, sr = sf.read(audio_path)

        # Stereo → Mono
        if len(audio.shape) > 1:
            audio = np.mean(
                audio,
                axis=1
            )

        audio = audio.astype(
            np.float32
        )

        # ------------------------------------
        # Expected ASVspoof sampling rate
        # ------------------------------------

        if sr != 16000:
            print(
                f"Unexpected sample rate: "
                f"{sr} | {file_name}"
            )

        # ------------------------------------
        # Keep first 5 seconds
        # ------------------------------------

        max_length = sr * MAX_SECONDS

        if len(audio) > max_length:
            audio = audio[:max_length]

        # ------------------------------------
        # Extract MFCC
        # ------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # ------------------------------------
        # Keep temporal information
        # ------------------------------------

        # MFCC shape:
        # (40, number_of_frames)

        # Pad or crop to fixed number
        # of frames.

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

            mfcc = mfcc[:, :MAX_FRAMES]

        # ------------------------------------
        # Add channel dimension
        # ------------------------------------

        mfcc = mfcc[..., np.newaxis]

        # Final shape:
        # (40, 400, 1)

        X.append(mfcc)

        # Bonafide = 0
        # Spoof = 1

        y.append(
            0 if label == "bonafide" else 1
        )

        processed += 1

        # ------------------------------------
        # Progress
        # ------------------------------------

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


# ============================================
# Convert to NumPy arrays
# ============================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)


# ============================================
# Save
# ============================================

print("\n================================")
print("FEATURE EXTRACTION COMPLETED")
print("================================")

print(
    "Processed:",
    len(X)
)

print(
    "Skipped:",
    skipped
)

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "Bonafide samples:",
    np.sum(y == 0)
)

print(
    "Spoof samples:",
    np.sum(y == 1)
)


np.save(
    OUTPUT_X,
    X
)

np.save(
    OUTPUT_Y,
    y
)


print("\nFeatures saved:")

print(
    OUTPUT_X
)

print(
    OUTPUT_Y
)