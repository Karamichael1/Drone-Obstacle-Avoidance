
import math
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# DWA and robot configuration (same as dynamicwindow2D.py)
class RobotType(Enum):
    circle = 0
    rectangle = 1

# Maze generator with difficulty-based obstacle spawning
class MazeGenerator:
    def __init__(self, width, height, start=(1, 1), goal=(46, 46)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.mazes = [self.generate_maze(difficulty * 2) for difficulty in range(1, 11)]

    def generate_maze(self, obstacle_count):
        maze = np.zeros((self.height, self.width))
        obstacles = []

        # Add boundary walls
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

        # Place random circular obstacles
        added_obstacles = 0
        while added_obstacles < obstacle_count:
            i = np.random.randint(1, self.height - 1)
            j = np.random.randint(1, self.width - 1)
            if maze[i, j] == 0 and (j, i) != self.start and (j, i) != self.goal:
                radius = 1.5
                obstacles.append((j, i, radius))
                maze[i, j] = 1
                added_obstacles += 1

        # Ensure start and goal are empty
        maze[self.start[1], self.start[0]] = 0
        maze[self.goal[1], self.goal[0]] = 0

        return maze, obstacles

    def get_maze(self, difficulty):
        if 1 <= difficulty <= 10:
            return self.mazes[difficulty - 1]
        else:
            raise ValueError("Difficulty must be between 1 and 10")

    @staticmethod
    def maze_to_obstacles(maze, obstacles):
        boundary_obstacles = []
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] == 1 and (i == 0 or i == maze.shape[0]-1 or j == 0 or j == maze.shape[1]-1):
                    boundary_obstacles.append((j, i, 0.5))  # Add boundary walls as obstacles
        return obstacles + boundary_obstacles
    def visualize_maze(self, maze, obstacles, ax):
        ax.imshow(maze, cmap='binary', origin='lower')

        # Plot the start and goal positions
        ax.plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Goal')

        # Plot the circular obstacles
        for (x, y, r) in obstacles:
            circle = plt.Circle((x, y), r, color='blue', fill=True)
            ax.add_artist(circle)

        ax.set_xlim(-1, self.width)
        ax.set_ylim(-1, self.height)
        ax.set_aspect('equal')
        ax.axis('off')

if __name__ == "__main__":
    maze_gen = MazeGenerator(50, 50)

    # Generate the mazes for all difficulty levels
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        maze, obstacles = maze_gen.get_maze(i + 1)
        maze_gen.visualize_maze(maze, obstacles, ax)
        ax.set_title(f"Maze {i + 1}")

    plt.tight_layout()
    plt.show()
