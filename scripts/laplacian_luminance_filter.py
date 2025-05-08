import os
import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from linear_laplaction_detection import detect_whitewater_foam
from PIL import Image

# ---------------------- Debug Saving Helper ----------------------

# def save_debug_image(array, step_name):
#     os.makedirs('debug_outputs', exist_ok=True)
#     if array.dtype != np.uint8:
#         array = (array * 255).astype(np.uint8)
#     img = Image.fromarray(array)
#     img.save(f'debug_outputs/{step_name}.png')
#     img.show(title=step_name)

# ---------------------- Flow Energy Computation ----------------------

def compute_flow_energy_field(image_np, enhance_contrast=True):
    """
    Computes a luminance-weighted water mask based on the binary water mask and grayscale image.
    Returns a float32 energy map the same size as the image.
    """
    # Get water mask
    water_mask = detect_whitewater_foam(image_np)
    # save_debug_image(water_mask, "09_water_mask_detected")

    # Convert to luminance
    luminance = rgb2gray(image_np)
    # save_debug_image(luminance, "10_raw_luminance")

    if enhance_contrast:
        # Rescale luminance contrast
        luminance = rescale_intensity(luminance, in_range=(0.2, 0.9), out_range=(0, 1))
        # save_debug_image(luminance, "11_contrast_stretched_luminance")

    # Manually initialize flow_energy as a 2D list of 0.0
    flow_energy = []
    for i in range(len(luminance)):
        row = []
        for j in range(len(luminance[0])):
            row.append(0.0)
        flow_energy.append(row)

    # Apply the luminance to water_masked pixels
    for i in range(len(luminance)):
        for j in range(len(luminance[0])):
            if water_mask[i][j]:
                flow_energy[i][j] = float(luminance[i][j])

    # flow_energy_np = np.array(flow_energy)
    # save_debug_image(flow_energy_np, "12_final_flow_energy_field")

    return flow_energy, water_mask

# ---------------------- Main Execution ----------------------

if __name__ == "__main__":
    image_path = "curious.jpg"
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    flow_energy, _ = compute_flow_energy_field(image_np)
