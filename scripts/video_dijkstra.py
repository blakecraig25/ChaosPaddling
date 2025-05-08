import os
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.graph import route_through_array
from laplacian_luminance_filter import compute_flow_energy_field
import math
from tqdm import tqdm

# ---------------------- Gaussian Helpers ----------------------

def gaussian_kernel(sigma, radius):
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=float)
    for u in range(-radius, radius + 1):
        for v in range(-radius, radius + 1):
            kernel[u + radius, v + radius] = math.exp(-(u**2 + v**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

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

# ---------------------- Cost Map + Drawing ----------------------

def cost_map(small_energy, small_mask, sigma=2):
    H, W = small_energy.shape
    cost = np.full((H, W), 1000.0, dtype=float)

    for i in range(H):
        for j in range(W):
            if small_mask[i, j]:
                cost[i, j] = 1.0 - small_energy[i, j]
    cost = gaussian_blur(cost, sigma)

    for i in range(H):
        for j in range(W):
            if not small_mask[i, j]:
                cost[i, j] = 1000.0
    return cost

def draw_route_on_image(image_np, route, thickness=1):
    overlay = (image_np * 255).astype(np.uint8).copy()
    H, W, _ = overlay.shape
    red_rgb = (216, 43, 39)

    for y, x in route:
        for dy in range(-thickness, thickness + 1):
            for dx in range(-thickness, thickness + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    overlay[ny, nx] = red_rgb
    return overlay

# ---------------------- Frame Processing ----------------------

def process_frame(frame, scale=1):
    image_np = frame / 255.0
    luminance = rgb2gray(image_np)

    energy_map, mask = compute_flow_energy_field((image_np * 255).astype(np.uint8))
    energy_map = np.array(energy_map)

    resized_shape = (int(image_np.shape[0] * scale), int(image_np.shape[1] * scale))
    small_img = resize(image_np, resized_shape, anti_aliasing=True)
    small_energy = resize(energy_map, resized_shape)
    small_mask = resize(mask.astype(float), resized_shape) > 0.5

    cost_result = cost_map(small_energy, small_mask)

    coords = np.argwhere(small_mask)
    if coords.size == 0:
        return (frame * 255).astype(np.uint8)

    start = tuple(coords[0])
    end = tuple(coords[-1])

    route, _ = route_through_array(cost_result, start, end, fully_connected=True)
    overlay = draw_route_on_image(small_img, np.array(route), thickness=5)

    # Resize overlay back to original frame size if needed
    overlay_resized = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return overlay_resized

# ---------------------- Video Processor ----------------------

def run_energy_path_route_on_video(video_path, output_path, scale=0.5, skip_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️ Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust output FPS based on selected frames
    output_fps = fps / skip_rate if skip_rate > 1 else fps

    codec = cv2.VideoWriter_fourcc(*'avc1')  # H.264 encoding
    out = cv2.VideoWriter(output_path, codec, output_fps, (width, height))

    print(f"📽 Writing ONLY augmented frames every {skip_rate} frames (output FPS: {output_fps:.2f})")

    frame_idx = 0

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay_rgb = process_frame(frame_rgb, scale=scale)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            out.write(overlay_bgr)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Done! Saved only augmented frames to: {output_path}")

# ---------------------- Run Example ----------------------

if __name__ == "__main__":
    input_video = "../test_videos/04_short_paddling.mp4"
    output_video = "../test_videos/04c_augmented_frames_only.mp4"
    run_energy_path_route_on_video(input_video, output_video, scale=0.5, skip_rate=2)
