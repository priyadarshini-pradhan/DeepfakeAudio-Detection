# DeepFake Audio Detection

A deep learning-based system for detecting whether an audio sample is
bonafide (real) or spoofed (fake).

## 1. Project Overview

This project detects synthetic or manipulated speech using the
ASVspoof 2019 Logical Access (LA) dataset.

The system uses MFCC-based audio features and a 2D Convolutional
Neural Network (CNN) to classify audio as:

- Bonafide / Real Audio
- Spoof / Fake Audio

A Flask web application provides an interface for uploading audio
and viewing the prediction result.

## 2. Dataset

Dataset used:

ASVspoof 2019 Logical Access (LA)

Training set:
- 25,380 audio samples
- 2,580 bonafide samples
- 22,800 spoof samples

Evaluation set:
- 71,237 audio samples
- 7,355 bonafide samples
- 63,882 spoof samples

The evaluation set contains attack categories A07–A19.

## 3. Methodology

The system follows this pipeline:

Audio Input
→ Audio Preprocessing
→ MFCC Feature Extraction
→ Fixed-Size MFCC Representation
→ 2D CNN
→ Classification
→ Flask Web Interface

### Audio Preprocessing

- Audio is converted to mono.
- The first five seconds are considered.
- The original 16 kHz sampling rate of the ASVspoof audio is retained.

### Feature Extraction

40 Mel-Frequency Cepstral Coefficients (MFCCs) are extracted.

The temporal dimension is padded or cropped to 400 frames.

Final feature representation:

40 × 400 MFCC matrix

### CNN Architecture

The final V2 model consists of:

- Conv2D: 32 filters
- Batch Normalization
- Max Pooling
- Conv2D: 64 filters
- Batch Normalization
- Max Pooling
- Conv2D: 128 filters
- Batch Normalization
- Max Pooling
- Global Average Pooling
- Dense: 128 neurons
- Dropout: 0.5
- Softmax output: 2 classes

## 4. Final Model

The final model is:

`Models/deepfake_cnn_v2.keras`

The V2 model preserves temporal MFCC information instead of averaging
the MFCC features across time.

## 5. Official Evaluation Results

| Metric | V1 | V2 |
|---|---:|---:|
| Accuracy | 80.50% | 89.44% |
| Precision (Spoof) | 97.06% | 99.61% |
| Recall (Spoof) | 80.70% | 88.57% |
| F1-score (Spoof) | 88.13% | 93.76% |
| ROC-AUC | 87.54% | 98.09% |
| EER | 20.11% | 7.51% |

The final V2 results are based on the official ASVspoof 2019 LA
evaluation set.

## 6. V2 Confusion Matrix

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 7,131 | 224 |
| Spoof | 7,302 | 56,580 |

## 7. Web Application

The project includes a Flask-based web application.

The application allows the user to:

1. Upload an audio file.
2. Process the audio automatically.
3. Extract MFCC features.
4. Run the trained CNN model.
5. Display REAL AUDIO or FAKE AUDIO.
6. Display the model's confidence score.
7. Display the audio waveform.

## 8. Project Structure

```text
Deep Fake N/
│
├── DataSet/
│   └── LA/
│
├── Features/
│   ├── X_v2.npy
│   └── y_v2.npy
│
├── Models/
│   └── deepfake_cnn_v2.keras
│
├── Results/
│   ├── Final_Graphs/
│   └── Final_Results/
│
├── app.py
├── extract_features_v2.py
├── train_v2.py
├── evaluate_v2.py
├── architecture_diagram.py
├── generate_final_graphs.py
└── README.md