import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
import os

# List of image paths
image_paths = [
    '../test_images/01_dark_test.jpg',
    '../test_images/02_bright_color.jpg',
    '../test_images/03_bright_difference.jpg',
    '../test_images/04_bright_brightest.jpg',
    '../test_images/05_bright_physics.jpg',
    '../test_images/06_bright_foam.jpg',
    '../test_images/07_bright_obstacles.jpg',
    '../test_images/08_dark_test2.jpg',
    '../test_images/09_dark_obstacles.jpg',
    '../test_images/11_bright_sky.jpg',
]

def plot_v_channel_histogram(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    hsv = rgb2hsv(image_np)
    val = hsv[:, :, 2].flatten()

    # Calculate mean brightness
    mean_val = np.mean(val)

    # Plot histogram
    hist, bins = np.histogram(val, bins=256, range=(0, 1), density=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.bar(bins[:-1], hist, width=0.004, align='edge', color='skyblue')
    ax.axvline(mean_val, color='purple', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3f}')
    ax.set_title(f"Brightness Histogram: {os.path.basename(image_path)}")
    ax.set_xlabel("Brightness (V channel)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    return mean_val

# Loop through all images and plot histograms
for path in image_paths:
    plot_v_channel_histogram(path)
