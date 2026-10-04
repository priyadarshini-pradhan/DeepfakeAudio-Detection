import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Output folder
output_folder = "Results/Final_Graphs"
os.makedirs(output_folder, exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(12, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis("off")

# Box drawing function
def draw_box(x, y, width, height, title, details):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        linewidth=1.8,
        edgecolor="black",
        facecolor="white"
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height - 0.35,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

    ax.text(
        x + width / 2,
        y + height / 2 - 0.15,
        details,
        ha="center",
        va="center",
        fontsize=10.5,
        linespacing=1.5
    )


# Title
ax.text(
    5,
    17.5,
    "DeepFake Audio Detection System Architecture",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold"
)

# Boxes
draw_box(
    1.5, 15.2, 7, 1.3,
    "1. Audio Input",
    "User uploads an audio file"
)

draw_box(
    1.5, 13.0, 7, 1.5,
    "2. Audio Preprocessing",
    "Convert stereo → mono\nFirst 5 seconds are selected\nOriginal 16 kHz sampling rate retained"
)

draw_box(
    1.5, 10.8, 7, 1.5,
    "3. MFCC Feature Extraction",
    "40 MFCC coefficients\nExtract temporal audio characteristics"
)

draw_box(
    1.5, 8.6, 7, 1.5,
    "4. Fixed-Size Feature Representation",
    "MFCC matrix: 40 × 400\nPadding / cropping applied to time dimension"
)

draw_box(
    1.5, 5.8, 7, 2.0,
    "5. 2D CNN Model",
    "Conv2D: 32 filters + Batch Normalization\n"
    "Conv2D: 64 filters + Batch Normalization\n"
    "Conv2D: 128 filters + Batch Normalization\n"
    "Max Pooling after convolution blocks"
)

draw_box(
    1.5, 3.4, 7, 1.6,
    "6. Classification Layer",
    "Global Average Pooling\nDense: 128 neurons\nDropout: 0.5\nSoftmax: 2 classes"
)

draw_box(
    1.5, 1.2, 7, 1.5,
    "7. Flask Web Interface",
    "REAL AUDIO / FAKE AUDIO\nConfidence score + waveform visualization"
)

# Arrows
arrow_props = dict(
    arrowstyle="->",
    linewidth=1.8,
    color="black"
)

arrow_positions = [
    (5, 15.2, 5, 14.5),
    (5, 13.0, 5, 12.3),
    (5, 10.8, 5, 10.1),
    (5, 8.6, 5, 7.8),
    (5, 5.8, 5, 5.0),
    (5, 3.4, 5, 2.7)
]

for x1, y1, x2, y2 in arrow_positions:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=arrow_props
    )

# Save diagram
output_path = os.path.join(
    output_folder,
    "system_architecture.png"
)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("=" * 40)
print("SYSTEM ARCHITECTURE GENERATED")
print("=" * 40)
print(f"Saved at: {output_path}")