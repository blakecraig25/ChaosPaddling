import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter

# ---------------------- Helper to Save and Show ----------------------

# def save_debug_image(array, step_name):
#     """Save and display intermediate image results."""
#     os.makedirs('debug_outputs', exist_ok=True)
#     if array.dtype != np.uint8:
#         array = (array * 255).astype(np.uint8)
#     img = Image.fromarray(array)
#     img.save(f'debug_outputs/{step_name}.png')
#     img.show(title=step_name)

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

def compute_linear_threshold_from_mean(image_np):
    hsv = rgb2hsv(image_np)
    val = hsv[:, :, 2].flatten()
    mean_val = np.mean(val)
    
    # Using both Equations
    # if mean_val >= 0.45:
    #     threshold = 2.857 * mean_val - .755
    # else:
    #     threshold = 0.245 * mean_val + 0.356
        
    # Using the linear regression equation only
    threshold = 0.245 * mean_val + 0.356
        
    
    threshold = np.clip(threshold, 0.35, 0.65)
    return threshold

def brightness_boost(gray, threshold):
    boosted = np.zeros_like(gray)
    H, W = gray.shape
    for i in range(H):
        for j in range(W):
            val = (gray[i, j] - threshold) / 0.6
            if val < 0:
                boosted[i, j] = 0
            elif val > 1:
                boosted[i, j] = 1
            else:
                boosted[i, j] = val

    # save_debug_image(boosted, "01_boosted_brightness")
    return boosted

def segment_foam(gray, threshold):
    boosted = brightness_boost(gray, threshold)
    laplace = laplacian(boosted)

    
    # normalized_laplace = (laplace - np.min(laplace)) / (np.max(laplace) - np.min(laplace))
    # save_debug_image(normalized_laplace, "02_laplacian_output")

    foam_mask = (laplace > 0.025) & (laplace < 0.07)
    foam_mask = binary_closing(foam_mask, disk(5))
    foam_mask = binary_opening(foam_mask, disk(3))
    foam_mask = remove_small_objects(foam_mask, min_size=500)

    # save_debug_image(foam_mask, "03_initial_foam_mask")
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

    # save_debug_image(rock_mask, "04_rock_mask")

    foam_mask[rock_mask] = False
    # save_debug_image(foam_mask, "05_foam_mask_after_rock_removal")
    return foam_mask

def segment_deep_water(hsv):
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    deep_water = (val < 0.55) & (sat < 0.5) & (hue > 0.45) & (hue < 0.70)

    # save_debug_image(deep_water, "06_deep_water_mask")
    return deep_water

def combine_and_clean_masks(foam_mask, deep_water_mask):
    combined = foam_mask | deep_water_mask
    filtered = median_filter(combined.astype(float), size=5) > 0.5
    cleaned = remove_small_objects(filtered, min_size=1000)

    # save_debug_image(cleaned, "07_combined_and_cleaned_mask")
    return cleaned

def extract_largest_region(mask):
    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return np.zeros_like(mask, dtype=bool)
    largest = max(regions, key=lambda r: r.area)
    final_mask = labeled == largest.label

    # save_debug_image(final_mask, "08_final_largest_region")
    return final_mask

# ---------------------- Combined Detection Pipeline ----------------------

def detect_whitewater_foam(image_np):
    gray = rgb2gray(image_np)
    hsv = rgb2hsv(image_np)
    threshold = compute_linear_threshold_from_mean(image_np)

    foam_mask = segment_foam(gray, threshold)
    foam_mask = remove_rock_regions(foam_mask, hsv)
    deep_water_mask = segment_deep_water(hsv)

    cleaned_mask = combine_and_clean_masks(foam_mask, deep_water_mask)
    final_mask = extract_largest_region(cleaned_mask)

    return final_mask

# ---------------------- Example Usage ----------------------

if __name__ == "__main__":
    image_path = "example.jpg"  # Replace with your actual path
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    final_mask = detect_whitewater_foam(image_np)

    overlay = image_np.copy()
    overlay[final_mask] = [255, 0, 0]
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    overlay_img.show(title="Final Overlay with Route")
    overlay_img.save('debug_outputs/final_overlay_route.png')
