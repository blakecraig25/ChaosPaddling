import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import sobel, gaussian_filter, variance
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

# Load the image
image_path = "exampe.jpg"  # Change to your image path
image = Image.open(image_path).convert("L")  # Convert to grayscale

# Convert to NumPy array and normalize
image_array = np.array(image) / 255.0

from skimage.color import rgb2hsv

# Load original RGB image for color processing
rgb_img = np.array(Image.open(image_path).convert("RGB")) / 255.0
hsv_img = rgb2hsv(rgb_img)
hue = hsv_img[:, :, 0]
sat = hsv_img[:, :, 1]
val = hsv_img[:, :, 2]

# Detect calm water as low saturation and medium-dark brightness
calm_water_color = ((sat < 0.2) & (val > 0.1) & (val < 0.4)).astype(int)




"""STEP 1: DETECT OBSTACLES"""
# Apply Gaussian smoothing to reduce noise
smoothed_image = gaussian_filter(image_array, sigma=1)

# Detect edges for object detection
edges_x = sobel(smoothed_image, axis=1)
edges_y = sobel(smoothed_image, axis=0)
gradient_magnitude = np.hypot(edges_x, edges_y)

# Compute local variance for texture detection (high variance = frothy water)
texture_variance = variance(smoothed_image, 5)

# Detect solid obstacles (rocks, trees)
obstacle_map = ((gradient_magnitude > 0.3) & (smoothed_image < 0.4)).astype(int)

# Detect navigable turbulent water (white frothy rapids)
frothy_water = ((texture_variance > 0.0001) & ((smoothed_image > 0.6).astype(int) | (smoothed_image < 0.001)).astype(int))

# Ensure frothy water is NOT an obstacle
obstacle_map[frothy_water == 1] = 0  # Remove from obstacle mask



"""STEP 2: ESTIMATE RIVER ANGLE & FLOW DIRECTION"""
# Compute gradients in x and y directions
flow_x = np.gradient(smoothed_image, axis=1)
flow_y = np.gradient(smoothed_image, axis=0)

# Compute the dominant flow direction (angle of river)
angle_map = np.arctan2(flow_y, flow_x)  # Angle in radians
avg_angle = np.mean(angle_map[np.abs(flow_y) > 0.05])  # Mean angle where flow is significant
river_angle_degrees = np.degrees(avg_angle)

# Define water flow mask: Areas with high brightness gradient are water
flow_map = ((flow_y > 0.02) & (smoothed_image > 0.5)).astype(int)

# Combine frothy water and regular water flow as green (navigable water)
# === Additional calm water and blue object detection ===
# Original luminance+variance calm water
calm_water_luma = ((texture_variance < 0.00005) & (smoothed_image > 0.05) & (smoothed_image < 0.25)).astype(int)

# Grayscale-based calm water: very dark and smooth
calm_water_backup = ((smoothed_image < 0.15) & (texture_variance < 0.00005)).astype(int)

# Extend HSV calm water thresholds
calm_water_color = ((sat < 0.35) & (val > 0.01) & (val < 0.45)).astype(int)

# Catch very dark, smooth regions as calm water
calm_water_very_dark = ((smoothed_image < 0.08) & (texture_variance < 0.00007)).astype(int)

# Merge calm water detection
calm_water = (
    (calm_water_luma == 1) |
    (calm_water_color == 1) |
    (calm_water_backup == 1) |
    (calm_water_very_dark == 1)
).astype(int)

# === Improved blue object detection (exclude blue water) ===

is_blue = ((hue > 0.55) & (hue < 0.75))
is_high_sat = sat > 0.4
is_bright_enough = val > 0.2
is_textured = texture_variance > 0.0001

# Only mark as obstacle if it's likely a raft/paddle (not smooth blue water)
blue_objects = (is_blue & is_high_sat & is_bright_enough & is_textured).astype(int)
obstacle_map[blue_objects == 1] = 1

# Final water map
full_flow_map = (flow_map | frothy_water | calm_water).astype(int)
full_flow_map = (flow_map | frothy_water | calm_water).astype(int)
full_flow_map = (flow_map | frothy_water | calm_water).astype(int)




"""STEP 3: BUILD GRAPH FOR ROUTE DETECTION"""
height, width = image_array.shape
size = height * width
graph = lil_matrix((size, size))

def index(y, x):
    return y * width + x

for y in range(height):
    for x in range(width):
        if obstacle_map[y, x] == 1:
            continue  # Skip obstacles
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring pixels
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and obstacle_map[ny, nx] == 0:
                weight = 1.0 - full_flow_map[ny, nx]  # Favor high flow areas
                graph[index(y, x), index(ny, nx)] = weight

# Convert graph to efficient format
graph = graph.tocsr()

# Define start (top center) and end (bottom center)
start, end = (10, width // 2), (height - 10, width // 2)
start_idx, end_idx = index(*start), index(*end)

# Compute shortest path using Dijkstra’s algorithm
distances, predecessors = dijkstra(graph, directed=True, indices=start_idx, return_predecessors=True)

# Reconstruct the optimal paddling path
path = []
current = end_idx
while current != start_idx and predecessors[current] != -9999:
    path.append(current)
    current = predecessors[current]

path_coords = []
for p in reversed(path):
    row = p // width
    col = p % width
    path_coords.append((row, col))



"""STEP 4: OVERLAY OBJECTS, FLOW, AND ROUTE"""
image_rgb = image.convert("RGB")
draw = ImageDraw.Draw(image_rgb)

# Draw obstacles (blue) – Solid rocks and trees only
for y in range(height):
    for x in range(width):
        if obstacle_map[y, x] == 1:
            draw.point((x, y), fill=(0, 0, 255))  # Blue overlay for obstacles

# Draw water flow (green) – Includes both regular flow and frothy rapids
for y in range(height):
    for x in range(width):
        if full_flow_map[y, x] == 1:
            draw.point((x, y), fill=(0, 255, 0))  # Green overlay for water flow

# Draw the computed red whitewater route
for y, x in path_coords:
    draw.ellipse((x-1, y-1, x+1, y+1), fill=(255, 0, 0))  # Thicker red route

# Save the processed image
output_path = "final_route_color_enhanced_other.png"
image_rgb.save(output_path)

print(f"Saved processed image: {output_path}")
