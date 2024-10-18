
import math
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# DWA and robot configuration (same as dynamicwindow2D.py)
class RobotType(Enum):
    circle = 0
    rectangle = 1

class Config:
    def __init__(self):
        self.max_speed = 1.5  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.3  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s]
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.45
        self.speed_cost_gain = 0.9
        self.obstacle_cost_gain = 1.0
        self.robot_radius = 1.0
        self.robot_type = RobotType.circle

def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)
    u = np.clip(u, -config.max_speed, config.max_speed)
    return u, trajectory

def calc_dynamic_window(x, config):
    Vs = [config.min_speed, config.max_speed, -config.max_yaw_rate, config.max_yaw_rate]
    return Vs

def calc_control_and_trajectory(x, dw, config, goal, ob):
    u = [0.0, 0.0]  # placeholder for control [v, yaw_rate]
    trajectory = np.array(x)  # Placeholder trajectory
    return u, trajectory

# Maze generator with difficulty-based obstacle spawning
class MazeGenerator:
    def __init__(self, width, height, start=(1, 1), goal=(46, 46)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal

        # Predefined mazes for different difficulty levels
        self.mazes = [self.generate_maze(difficulty * 2) for difficulty in range(1, 11)]

    def generate_maze(self, obstacle_count):
        maze = np.zeros((self.height, self.width))
        obstacles = []

        # Add boundary walls (1s along the edges)
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1

        # Place random circular obstacles inside the maze grid
        added_obstacles = 0
        while added_obstacles < obstacle_count:
            i = np.random.randint(1, self.height - 1)
            j = np.random.randint(1, self.width - 1)
            if maze[i, j] == 0 and (j, i) != self.start and (j, i) != self.goal:
                radius = 1.0  # Adding a fixed radius for each obstacle
                obstacles.append((j, i, radius))  # Store position and radius of obstacles
                maze[i, j] = 1
                added_obstacles += 1

        # Ensure the start and goal points are empty
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
        return obstacles

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
