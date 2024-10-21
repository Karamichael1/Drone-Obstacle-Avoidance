import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from heapq import heappush, heappop

from dynamicwindow2D import Config, RobotType, dwa_control, motion, plot_arrow
from customastar import CustomAStar, CustomAStarAgent

show_animation = True

class AStarDWAAgent:
    def __init__(self, static_obstacles, resolution, robot_radius, max_speed):
        self.static_obstacles = static_obstacles
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.custom_astar = CustomAStarAgent(resolution, robot_radius)
        self.dwa_config = Config()
        self.dwa_config.robot_type = RobotType.circle
        self.dwa_config.robot_radius = robot_radius
        self.dwa_config.max_speed = max_speed
        self.collision_check_distance = robot_radius * 2

    def move_to_goal(self, start, goal, dynamic_obstacles):
        x = np.array(list(start) + [math.pi / 8.0, 0.0, 0.0])
        trajectory = np.array([x])
        
        path = self.custom_astar.plan_path(start, goal, self.static_obstacles)
        if not path:
            print("No initial CustomA* path found. Defaulting to DWA.")
            path = None
        else:
            rx, ry = zip(*path)
            target_ind = 0
        
        time = 0
        in_dwa_mode = path is None
        
        while True:
            current_obstacles = self.update_obstacles(time, dynamic_obstacles)
            dwa_obstacles = np.array([[obs[0], obs[1]] for obs in current_obstacles])
            
            if path and target_ind < len(rx):
                local_goal = [rx[target_ind], ry[target_ind]]
            else:
                local_goal = goal
            
            if in_dwa_mode or self.check_collision(x[:2], current_obstacles):
                in_dwa_mode = True
                u, predicted_trajectory = dwa_control(x, self.dwa_config, local_goal, dwa_obstacles)
            else:
                angle = math.atan2(local_goal[1] - x[1], local_goal[0] - x[0])
                speed = min(self.dwa_config.max_speed, math.hypot(local_goal[0] - x[0], local_goal[1] - x[1]))
                u = np.array([speed, angle - x[2]])
                predicted_trajectory = np.array([x[:2], local_goal])
            
            x = motion(x, u, self.dwa_config.dt)
            trajectory = np.vstack((trajectory, x))
            
            dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
            if dist_to_goal <= self.resolution:
                print("Goal reached!")
                break
            
            if path:
                dist_to_target = math.hypot(x[0] - local_goal[0], x[1] - local_goal[1])
                if dist_to_target <= self.resolution:
                    if target_ind < len(rx) - 1:
                        target_ind += 1
                    else:
                        # Reached end of CustomA* path, switch to DWA
                        in_dwa_mode = True
                        path = None
            
            if in_dwa_mode and not self.check_collision(x[:2], current_obstacles):
                # Try to find a new CustomA* path
                new_path = self.custom_astar.plan_path(x[:2], goal, current_obstacles)
                if new_path:
                    path = new_path
                    rx, ry = zip(*path)
                    target_ind = 0
                    in_dwa_mode = False
                    print("Found new CustomA* path. Switching back to CustomA*.")
            
            if show_animation:
                self.plot_state(x, goal, rx if path else None, ry if path else None, predicted_trajectory, current_obstacles, in_dwa_mode)
            
            time += self.dwa_config.dt
        
        return trajectory

    def check_collision(self, position, obstacles):
        for obs in obstacles:
            if math.hypot(position[0] - obs[0], position[1] - obs[1]) <= self.collision_check_distance:
                return True
        return False

    def update_obstacles(self, time, dynamic_obstacles):
        current_obstacles = self.static_obstacles.copy()
        for obs in dynamic_obstacles:
            x = obs['x'] + obs['vx'] * time
            y = obs['y'] + obs['vy'] * time
            current_obstacles.append((x, y, obs['radius']))  
        return current_obstacles

    def plot_state(self, x, goal, rx, ry, predicted_trajectory, obstacles, in_dwa_mode):
        plt.cla()
        plt.plot(x[0], x[1], "xr")
        plt.plot(goal[0], goal[1], "xb")
        if rx and ry:
            plt.plot(rx, ry, "-r")  # Plot the entire CustomA* path
        
        self.plot_obstacles(obstacles)
        
        plot_arrow(x[0], x[1], x[2])
        plt.axis("equal")
        plt.grid(True)
        plt.title(f"AStarDWAAgent Navigation ({'DWA' if in_dwa_mode else 'CustomA*'} mode)")
        plt.pause(0.0001)

    def plot_obstacles(self, obstacles):
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], fill=True)
            plt.gca().add_artist(circle)

def create_maze(width, height, obstacle_ratio):  
    maze = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height-1 or j == 0 or j == width-1:
                maze[i, j] = 1 
            elif np.random.rand() < obstacle_ratio:
                maze[i, j] = 1 
    return maze

def maze_to_obstacles(maze):
    obstacles = []
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 1:
                obstacles.append((j, i, 0.5)) 
    return obstacles