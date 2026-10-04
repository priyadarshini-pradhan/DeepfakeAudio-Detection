import os
import matplotlib.pyplot as plt
import numpy as np

# Create output folder
os.makedirs("Results/Final_Graphs", exist_ok=True)


# ============================================================
# 1. V1 vs V2 PERFORMANCE COMPARISON
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

v1 = [
    80.50,
    97.06,
    80.70,
    88.13,
    87.54
]

v2 = [
    89.44,
    99.61,
    88.57,
    93.76,
    98.09
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width / 2, v1, width, label="V1")
plt.bar(x + width / 2, v2, width, label="V2")

plt.xticks(x, metrics)
plt.ylabel("Score (%)")
plt.title("V1 vs V2 Model Performance")
plt.ylim(0, 105)
plt.legend()

plt.tight_layout()

plt.savefig(
    "Results/Final_Graphs/v1_vs_v2_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. V2 CONFUSION MATRIX
# ============================================================

cm = np.array([
    [7131, 224],
    [7302, 56580]
])

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("V2 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Bonafide", "Spoof"]
)

plt.yticks(
    [0, 1],
    ["Bonafide", "Spoof"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "Results/Final_Graphs/v2_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. V2 ATTACK-WISE ROC-AUC
# ============================================================

attacks = [
    "A07", "A08", "A09", "A10",
    "A11", "A12", "A13", "A14",
    "A15", "A16", "A17", "A18", "A19"
]

auc_scores = [
    0.9997,
    0.9928,
    0.9999,
    0.9987,
    0.9995,
    0.9992,
    0.9985,
    0.9978,
    0.9982,
    0.9993,
    0.8538,
    0.9246,
    0.9900
]

plt.figure(figsize=(11, 6))

plt.bar(
    attacks,
    auc_scores
)

plt.xlabel("Attack Category")
plt.ylabel("ROC-AUC")
plt.title("V2 Attack-wise ROC-AUC")

plt.ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    "Results/Final_Graphs/v2_attack_wise_auc.png",
    dpi=300
)

plt.close()


print("================================")
print("FINAL GRAPHS GENERATED")
print("================================")

print("Saved inside:")
print("Results/Final_Graphs/")

print()
print("Files created:")
print("1. v1_vs_v2_performance.png")
print("2. v2_confusion_matrix.png")
print("3. v2_attack_wise_auc.png")