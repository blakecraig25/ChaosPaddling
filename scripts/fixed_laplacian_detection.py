import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter

# ---------------------- Core Math Operations ----------------------

def laplacian(image):
    H, W = image.shape
    laplace = np.zeros((H, W), dtype=float)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            center = image[i, j]
            top = image[i - 1, j]
            bottom = image[i + 1, j]
            left = image[i, j - 1]
            right = image[i, j + 1]
            laplace[i, j] = -4 * center + top + bottom + left + right
    return laplace

def brightness_boost(gray):
    H, W = gray.shape
    boosted = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            val = (gray[i, j] - 0.4) / 0.6
            if val < 0:
                boosted[i, j] = 0
            elif val > 1:
                boosted[i, j] = 1
            else:
                boosted[i, j] = val

    return boosted

# ---------------------- Feature Segmentation ----------------------

def segment_foam(gray):
    boosted = brightness_boost(gray)
    laplace = laplacian(boosted)
    foam_mask = (laplace > 0.025) & (laplace < 0.07)
    foam_mask = binary_closing(foam_mask, disk(5))
    foam_mask = binary_opening(foam_mask, disk(3))
    foam_mask = remove_small_objects(foam_mask, min_size=500)
    return foam_mask

def remove_rock_regions(foam_mask, hsv):
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    height_quarter = hsv.shape[0] // 4
    rock_mask = (
        (val > 0.65) &
        (sat > 0.2) &
        (hue > 0.03) & (hue < 0.12)
    )
    rock_mask[:height_quarter, :] = rock_mask[:height_quarter, :]
    foam_mask[rock_mask] = False
    return foam_mask

def segment_deep_water(hsv):
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (
        (val < 0.55) &
        (sat < 0.5) &
        (hue > 0.45) & (hue < 0.70)
    )

# ---------------------- Post-Processing ----------------------

def combine_and_clean_masks(foam_mask, deep_water_mask):
    combined = foam_mask | deep_water_mask
    filtered = median_filter(combined.astype(float), size=5) > 0.5
    cleaned = remove_small_objects(filtered, min_size=1000)
    return cleaned

def extract_largest_region(mask):
    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return np.zeros_like(mask, dtype=bool)
    largest = max(regions, key=lambda r: r.area)
    return labeled == largest.label

# ---------------------- Combined Water Detection ----------------------

def detect_whitewater_foam(image_np):
    gray = rgb2gray(image_np)
    hsv = rgb2hsv(image_np)

    foam_mask = segment_foam(gray)
    foam_mask = remove_rock_regions(foam_mask, hsv)
    deep_water_mask = segment_deep_water(hsv)

    cleaned_mask = combine_and_clean_masks(foam_mask, deep_water_mask)
    final_mask = extract_largest_region(cleaned_mask)

    return final_mask

# ---------------------- Main Execution ----------------------

if __name__ == "__main__":
    image_path = "example.jpg"  # Replace with your image file
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # -- Test Case: Deep Water Only --
    hsv = rgb2hsv(image_np)
    deep_water_mask = segment_deep_water(hsv)

    overlay = image_np.copy()
    overlay[deep_water_mask] = [0, 255, 255]  # Cyan overlay for deep water

    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    overlay_img.show()
