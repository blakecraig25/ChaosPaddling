import numpy as np
from PIL import Image

# Load the RGB image
image_path = "example.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
rgb_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

# Step 1: Compute luminance (grayscale)
luminance = np.dot(rgb_array[..., :3], [0.299, 0.587, 0.114])[..., None]  # Shape: (H, W, 1)

# Step 2: Invert luminance
inverted_luminance = 1.0 - luminance

# Step 3: Isolate the blue channel
blue_channel = rgb_array[..., 2:3]  # Keep blue channel only
blue_only = np.concatenate([
    np.zeros_like(blue_channel),   # Red = 0
    np.zeros_like(blue_channel),   # Green = 0
    blue_channel                   # Blue retained
], axis=-1)

# Step 4: Compute adaptive threshold
avg_luminance = np.mean(luminance)
avg_blue = np.mean(blue_channel)
threshold = (avg_luminance + avg_blue) / 3  # Dynamic threshold between blue and brightness

# Step 5: Blend based on threshold
blue_strength = (blue_channel > threshold).astype(float)
output = (1 - blue_strength) * inverted_luminance + blue_strength * blue_only
output = np.clip(output, 0, 1)

# Step 6: Convert to displayable image
output_image = Image.fromarray((output * 255).astype(np.uint8))
output_image.save("inverted_luminance_with_blue_highlight.png")
output_image.show()
