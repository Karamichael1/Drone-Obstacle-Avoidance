from PIL import Image


image1_path = "nice example/astardwa_maze_1.png"
image2_path = "nice example/customastar_maze_1.png"
image3_path = "nice example/dwa_maze_1.png"

import matplotlib.pyplot as plt

# Load the images as arrays for overlaying paths
import matplotlib.image as mpimg

# Load images for plotting
img1 = mpimg.imread(image1_path)
img2 = mpimg.imread(image2_path)
img3 = mpimg.imread(image3_path)

# Plot all paths on a single figure with different colors
plt.figure(figsize=(8, 8))

# Display the maze background using one of the images as they are all of the same background
plt.imshow(img1)

# Plot the paths in different colors by overlaying each image
plt.plot([], [], color="blue", label="astardwa_maze_1")  # Blue path for astardwa_maze_1
plt.plot([], [], color="green", label="customastar_maze_1")  # Green path for customastar_maze_1
plt.plot([], [], color="red", label="dwa_maze_1")  # Red path for dwa_maze_1

# Extract the color-coded paths from each image and overlay
plt.imshow(img1[:, :, :3], alpha=0.6)  # First path in blue tint
plt.imshow(img2[:, :, :3], alpha=0.4)  # Second path in green tint
plt.imshow(img3[:, :, :3], alpha=0.3)  # Third path in red tint

# Add legend and axis details
plt.legend(loc="upper right")
plt.axis("off")

# Show the combined plot
plt.show()
