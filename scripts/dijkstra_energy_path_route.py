import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.graph import route_through_array
from laplacian_luminance_filter import compute_flow_energy_field
import math

# ---------------------- Debug Saving Helper ----------------------

# def save_debug_image(array, step_name):
#     os.makedirs('debug_outputs', exist_ok=True)
#     if array.dtype != np.uint8:
#         array = (array * 255).astype(np.uint8)
#     img = Image.fromarray(array)
#     img.save(f'debug_outputs/{step_name}.png')
#     img.show(title=step_name)

# ---------------------- Gaussian Kernel ----------------------

def gaussian_kernel(sigma, radius):
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=float)
    for u in range(-radius, radius + 1):
        for v in range(-radius, radius + 1):
            kernel[u + radius, v + radius] = math.exp(-(u**2 + v**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

# ---------------------- Gaussian Blur ----------------------

def gaussian_blur(image, sigma):
    radius = int(3 * sigma)
    kernel = gaussian_kernel(sigma, radius)
    H, W = image.shape
    blurred = np.zeros_like(image)

    for i in range(H):
        for j in range(W):
            val = 0.0
            for u in range(-radius, radius + 1):
                for v in range(-radius, radius + 1):
                    ii = i + u
                    jj = j + v
                    if 0 <= ii < H and 0 <= jj < W:
                        val += image[ii, jj] * kernel[u + radius, v + radius]
            blurred[i, j] = val
    return blurred

# ---------------------- Cost Map Construction ----------------------

def cost_map(small_energy, small_mask, sigma=2):
    H, W = small_energy.shape
    cost_map = np.full((H, W), 1000.0, dtype=float)

    # Invert flow energy in water regions
    for i in range(H):
        for j in range(W):
            if small_mask[i, j]:
                cost_map[i, j] = 1.0 - small_energy[i, j]

    # save_debug_image(cost_map, "13_inverted_energy_cost_map")

    # Apply manual Gaussian blur
    cost_map = gaussian_blur(cost_map, sigma)

    # save_debug_image(cost_map, "14_blurred_cost_map")

    # Reinforce high cost in non-water areas
    for i in range(H):
        for j in range(W):
            if not small_mask[i, j]:
                cost_map[i, j] = 1000.0

    # save_debug_image(cost_map, "15_final_cost_map_with_blocks")
    return cost_map

# ---------------------- Route Finding ----------------------

def find_route(cost_map, small_mask):
    coords = np.argwhere(small_mask)
    start_y, end_y = np.min(coords[:, 0]), np.max(coords[:, 0])
    mid_x = small_mask.shape[1] // 2
    start = (start_y, mid_x)
    end = (end_y, mid_x)

    route, _ = route_through_array(cost_map, start, end, fully_connected=True)
    return np.array(route)

# ---------------------- Overlay Drawing ----------------------

def draw_route_on_image(image_np, route, thickness=1):
    overlay = (image_np * 255).astype(np.uint8).copy()
    H, W, _ = overlay.shape
    red_rgb = (216, 43, 39)  # Hex #d82b27

    for y, x in route:
        for dy in range(-thickness, thickness + 1):
            for dx in range(-thickness, thickness + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    overlay[ny, nx] = red_rgb
    return overlay

# ---------------------- Main Pipeline ----------------------

def run_energy_path_route(image_path, scale=1):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    luminance = rgb2gray(image_np)

    # Get flow energy + mask
    energy_map, mask = compute_flow_energy_field(image_np)
    energy_map = np.array(energy_map)

    # Downscale
    resized_shape = (int(image_np.shape[0]*scale), int(image_np.shape[1]*scale))
    small_img = resize(image_np, resized_shape, anti_aliasing=True)
    small_energy = resize(energy_map, resized_shape)
    small_mask = resize(mask.astype(float), resized_shape) > 0.5

    # save_debug_image(small_energy, "16_small_resized_energy_map")
    # save_debug_image(small_mask, "17_small_resized_water_mask")

    # Build cost map
    cost_map_result = cost_map(small_energy, small_mask)

    # Use the first nonzero point in the mask as the start
    coords = np.argwhere(small_mask)
    if coords.size == 0:
        print("No valid water mask detected.")
        return
    start = tuple(coords[0])
    end = tuple(coords[-1])

    # Compute route
    indices, _ = route_through_array(cost_map_result, start, end, fully_connected=True)
    route = np.array(indices)

    # Draw and show route
    overlay = draw_route_on_image(small_img, route, thickness=5)
    final_overlay = Image.fromarray(overlay)
    final_overlay.show(title="Final River Route")
    # final_overlay.save('debug_outputs/18_final_river_route_overlay.png')

# ---------------------- Entry Point ----------------------

if __name__ == "__main__":
    run_energy_path_route("../test_images/06_bright_foam.jpg")
