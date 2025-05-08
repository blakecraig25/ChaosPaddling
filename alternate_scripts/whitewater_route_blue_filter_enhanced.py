import numpy as np
from PIL import Image
from scipy.ndimage import variance
from scipy.ndimage import generic_filter

# Load the RGB image
image_path = "example.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
rgb_array = np.array(image) / 255.0  # Normalize to [0, 1]

# Step 1: Extract RGB channels
R, G, B = rgb_array[..., 0:1], rgb_array[..., 1:2], rgb_array[..., 2:3]

# Step 2: Compute adaptive blue threshold
avg_blue = np.mean(B)
avg_rgb = np.mean(rgb_array)
blue_thresh = (avg_blue + avg_rgb) / 2.0
blue_mask = (B > blue_thresh).astype(float)

# Step 3: Detect dark gray water (non-blue)
gray_std = np.std(rgb_array, axis=2, keepdims=True)
dark_gray_mask = ((np.mean(rgb_array, axis=2, keepdims=True) < 0.3) & (gray_std < 0.05)).astype(float)

# Step 4: Merge water masks
water_mask = np.maximum(blue_mask, dark_gray_mask)

# Step 5: Detect rocks via texture
gray_image = np.mean(rgb_array, axis=2)

def local_variance(values):
    return np.var(values)

texture_map = generic_filter(gray_image, local_variance, size=5)

rock_mask = (texture_map > 0.005).astype(float)[..., None]  # Match shape (H, W, 1)

# Step 6: Remove rocks from water
water_mask_cleaned = np.clip(water_mask - rock_mask, 0, 1)

highlight_rgb = 0.2 * rgb_array + 0.8 * np.concatenate([
    np.zeros_like(B),
    np.zeros_like(B),
    np.ones_like(B)
], axis=-1)

# Step 8: Final blend
output = (1- water_mask_cleaned) * rgb_array + water_mask_cleaned * highlight_rgb
output = np.clip(output, 0, 1)

# Step 9: Convert to image and show
output_image = Image.fromarray((output * 255).astype(np.uint8))
output_image.show()
