import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def linear_contrast_enhancement_pillow(image, percent=0.02):
    """
    Linearly enhance contrast using only Pillow.
    """
    if image.mode != 'L':
        raise ValueError("Image must be in grayscale ('L') mode")

    pixels = list(image.getdata())
    width, height = image.size
    total_pixels = width * height

    # Histogram
    histogram = [0] * 256
    for px in pixels:
        histogram[px] += 1

    # Cumulative histogram
    cumulative = []
    cum_sum = 0
    for count in histogram:
        cum_sum += count
        cumulative.append(cum_sum)

    # Quantile bounds
    lower_thresh = int(total_pixels * percent)
    upper_thresh = int(total_pixels * (1 - percent))

    low_val = next(i for i, c in enumerate(cumulative) if c >= lower_thresh)
    high_val = next(i for i in reversed(range(256)) if cumulative[i] <= upper_thresh)

    if high_val == low_val:
        high_val = low_val + 1

    # Contrast stretch
    stretched_pixels = []
    for px in pixels:
        if px <= low_val:
            stretched_px = 0
        elif px >= high_val:
            stretched_px = 255
        else:
            stretched_px = int((px - low_val) * 255 / (high_val - low_val))
        stretched_pixels.append(stretched_px)

    new_image = Image.new('L', (width, height))
    new_image.putdata(stretched_pixels)
    return new_image

def auto_water_color_mask(hsv, p=0.02):
    """
    Auto-adjusted water color mask using HSV histogram quantiles.
    """
    h, s, v = cv2.split(hsv)
    flat_h = h.flatten()
    flat_s = s.flatten()

    h_low, h_high = np.quantile(flat_h, [p, 1 - p])
    s_low, s_high = np.quantile(flat_s, [p, 1 - p])

    h_low = max(80, h_low)
    h_high = min(150, h_high)
    s_low = max(10, s_low)
    s_high = min(255, s_high)

    lower_bound = np.array([h_low, s_low, 30], dtype=np.uint8)
    upper_bound = np.array([h_high, s_high, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask

def texture_filter(gray_image):
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    smoothed = cv2.GaussianBlur(magnitude, (7, 7), 0)
    _, low_texture_mask = cv2.threshold(smoothed.astype(np.uint8), 20, 255, cv2.THRESH_BINARY_INV)
    return low_texture_mask

def combine_masks(mask1, mask2):
    combined = cv2.bitwise_and(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def highlight_mask(image, mask):
    result = image.copy()
    result[mask > 0] = [0, 0, 255]  # Red highlight for water
    return result

def main():
    image_path = "example.jpg"
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print("❌ Could not load image.")
        return

    image_cv = cv2.resize(image_cv, (640, 480))
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Convert to grayscale and enhance with Pillow
    gray_pil = image_pil.convert("L")
    enhanced_pil = linear_contrast_enhancement_pillow(gray_pil)

    # Convert enhanced PIL image back to OpenCV format
    enhanced_gray = np.array(enhanced_pil)

    # HSV for color masking
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    color_mask = auto_water_color_mask(hsv, p=0.02)
    texture_mask = texture_filter(enhanced_gray)
    final_mask = combine_masks(color_mask, texture_mask)

    highlighted = highlight_mask(image_cv, final_mask)

    # Show results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Water Mask")
    plt.imshow(final_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Detected Water (Red)")
    plt.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
