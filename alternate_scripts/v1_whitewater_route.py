import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import sobel, gaussian_filter, variance
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

# Load the image
image_path = "example.jpg"  # Change to your image path
image = Image.open(image_path).convert("L")  # Convert to grayscale

# Convert to NumPy array and normalize
image_array = np.array(image) / 255.0



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
full_flow_map = (flow_map | frothy_water).astype(int)




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
output_path = "final_route.png"
image_rgb.save(output_path)

print(f"Saved processed image: {output_path}")
