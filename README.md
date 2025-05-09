# [Chaos Paddling 🛶](https://github.com/blakecraig25/ChaosPaddling)

## Whitewater Navigation Path Detection

Whitewater rapids are thrilling, unpredictable, and often unforgiving. Behind the turbulence, there’s always a best line — the path of least resistance that threads between rocks, foam, and swirling currents. This project visualizes that line automatically.

Using still images or video of rapids, this software analyzes the terrain and **overlays a red navigation route** that follows the natural flow, avoiding hazards and highlighting the safest descent.

---

### What This Tool Does

This software processes each frame of a whitewater rapid, identifying safe water regions and detecting key obstacles like rocks, trees, and foam. It then generates a **cost map**, where every pixel is rated based on how safe or risky it is to travel through.

With that map, it applies **Dijkstra’s algorithm** to trace the optimal route from the top of the image to the bottom — the digital equivalent of a paddler scouting the rapid.

The final result is a clear, visual overlay on the original image or video — a red line showing the most navigable path.

---

### How It Works

1. **Preprocessing**  
   Converts input to grayscale and enhances contrast to normalize lighting differences across scenes.

2. **Water & Obstacle Detection**  
   Uses a custom luminance + Laplacian edge filter to detect water regions while avoiding foam, rocks, and land.

3. **Energy Map Construction**  
   Calculates a flow energy field, assigning lower cost to smoother water and higher cost to obstacles.

4. **Pathfinding**  
   Runs Dijkstra’s algorithm to compute the lowest-cost path from the top to bottom of the rapid.

5. **Output Rendering**  
   Draws the computed route in red on the original image, or frame-by-frame for video output.

---

### Example Inputs & Outputs

**Input Image**  
*An unprocessed frame of a whitewater rapid*  
![Input](test_images/13_dark_test4.jpg)

**Output Image with Imposed Route**  
*The route traced from in red*  
![Output](Image_Results/13_image_result.jpg)

**Sample Output Video**  
Each frame of the video includes the computed navigation path.  
<video src="test_videos/04c_augmented_frames_only.mp4" width="320" height="240" controls></video>

The video is too large, so going to test_videos/04c_augmented_frames_only.mp4 and downloading it will make the video available.

There are also more results within the [Image_Results](Image_Results) folder.

---

### File Structure

```
whitewater_navigation/
├── video_dijkstra.py             # Main video pipeline
├── dijkstra_energy_path_route.py # For single image processing
├── laplacian_luminance_filter.py # Energy map and water detection logic
├── examples/                     # Input/output media
├── utils/                        # Helper functions
└── README.md
```

---

### Why It Matters

By automating route detection in whitewater, this project creates a foundation for safer scouting, smarter river analysis, and even future applications like augmented reality paddling aids. It brings image processing and river logic together — to help us read the water, one frame at a time.
