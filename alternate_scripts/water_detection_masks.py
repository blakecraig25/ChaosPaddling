
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the uploaded image
image_path = "example.jpg"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Resize for quicker processing
resized = cv2.resize(image_np, (640, 480))
bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# 1. Estimate water mask based on color (broad range for green/blue/white water)
lower_water = np.array([40, 20, 40])   # loose bounds to include whitewater & deep
upper_water = np.array([140, 255, 255])
color_mask = cv2.inRange(hsv, lower_water, upper_water)

# 2. Estimate "whitewater" with high brightness and low saturation (foamy areas)
whitewater_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))

# 3. Estimate obstacles via high texture regions (Sobel edges)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges = np.sqrt(sobelx**2 + sobely**2)
obstacles = cv2.threshold(edges.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)[1]

# 4. Combine water and whitewater, remove obstacles
navigable_water = cv2.bitwise_and(color_mask, cv2.bitwise_not(obstacles.astype(np.uint8)))

# 5. Visualize all masks
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(color_mask, cmap='gray')
axes[1].set_title("Water Mask")
axes[2].imshow(whitewater_mask, cmap='gray')
axes[2].set_title("Whitewater (Fast Flow)")
axes[3].imshow(navigable_water, cmap='gray')
axes[3].set_title("Navigable Water (Est.)")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
