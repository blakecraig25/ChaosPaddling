import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2hsv

# --------- Step 1: Ordered image paths ----------
image_paths = [
    '../test_images/08_dark_test2.jpg',
    '../test_images/07_bright_obstacles.jpg',
    '../test_images/11_bright_sky.jpg',
    '../test_images/01_dark_test.jpg',
    '../test_images/05_bright_physics.jpg',
    '../test_images/09_dark_obstacles.jpg',
    '../test_images/04_bright_brightest.jpg',
    '../test_images/06_bright_foam.jpg',
    '../test_images/03_bright_difference.jpg',
    '../test_images/02_bright_color.jpg'
]

# --------- Step 2: Compute mean brightness ----------
mean_brightness = []
for path in image_paths:
    image = Image.open(path).convert('RGB')
    image_np = np.array(image) / 255.0
    hsv = rgb2hsv(image_np)
    val_channel = hsv[:, :, 2]
    mean_val = np.mean(val_channel)
    mean_brightness.append(mean_val)

# --------- Step 3: Ideal thresholds ----------
ideal_thresholds = [
    0.46,  # dark_test2
    0.61,  # bright_obstacles
    0.58,  # bright_sky
    0.48,  # dark_test
    0.42,  # bright_physics
    0.43,  # dark_obstacles
    0.41,  # brightest
    0.57,  # bright_foam
    0.55,  # bright_difference
    0.44   # bright_color
]

# --------- Step 4: Plot ----------
x_vals = np.linspace(0, 1, 100)
y_vals = 0.245 * x_vals + 0.356

# Find bounds of the "valid" region
xmin, xmax = min(mean_brightness) - .025, max(mean_brightness) + .025
ymin, ymax = min(ideal_thresholds) - .025, max(ideal_thresholds) + .025

plt.figure(figsize=(8, 6))

# Fill gray outside the valid region
plt.axhspan(0, ymin, color='gray')
plt.axhspan(ymax, 1, color='gray')
plt.axvspan(0, xmin, color='gray')
plt.axvspan(xmax, 1, color='gray')

# Plot points and regression line
plt.scatter(mean_brightness, ideal_thresholds, color='blue', marker='.', label='Data Points')
plt.plot(x_vals, 0.245 * x_vals + 0.356, color='red', label='Fit Line: y = 0.245x + 0.356')
plt.plot(x_vals, 2.857 * x_vals - 0.755, color='green', label='Fit Line: y = 0.2.857x - 0.755')

# Set axes limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# Labels, grid, legend
plt.xlabel('Mean Brightness (HSV V-channel)')
plt.ylabel('Ideal Threshold')
plt.title('Brightness vs. Ideal Threshold Regression')
plt.grid(False)
plt.legend()
plt.show()
