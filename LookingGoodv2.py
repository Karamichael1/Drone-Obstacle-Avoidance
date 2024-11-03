import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from heapq import heappush, heappop

from dynamicwindow2D import Config, RobotType, dwa_control, motion, plot_arrow
from customastar import CustomAStar, CustomAStarAgent

show_animation = True



class AStarDWAAgent:
    def __init__(self, static_obstacles, resolution, robot_radius, max_speed, map_width, map_height):
        self.static_obstacles = static_obstacles
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.custom_astar = CustomAStarAgent(resolution, robot_radius, map_width, map_height)
        self.dwa_config = Config()
        self.dwa_config.robot_type = RobotType.circle
        self.dwa_config.robot_radius = robot_radius
        self.dwa_config.max_speed = max_speed
        self.replan_interval = 5  # Replan 
        self.prediction_horizon = 3.0  # Look ahead 2 seconds
        self.dwa_activation_distance = robot_radius * 1  # Distance to switch to DWA
        self.map_width = map_width
        self.map_height = map_height

    def predict_obstacle_positions(self, dynamic_obstacles, time):
        """Predict positions of dynamic obstacles at a given time."""
        predicted = []
        for obs in dynamic_obstacles:
            if 'pos' in obs and 'vel' in obs:
                x = float(obs['pos'][0] + obs['vel'][0] * time)
                y = float(obs['pos'][1] + obs['vel'][1] * time)
            else:
                x = float(obs['x'] + obs['vx'] * time)
                y = float(obs['y'] + obs['vy'] * time)
            predicted.append((x, y, obs['radius']))
        return predicted

    def check_path_safety(self, path, dynamic_obstacles, current_time):
        """Check if the current path is safe considering dynamic obstacles."""
        if not path:
            return False
        
        check_points = np.linspace(0, self.prediction_horizon, 10)
        for t in check_points:
            predicted_obstacles = self.predict_obstacle_positions(dynamic_obstacles, current_time + t)
            path_point_index = min(int(t / self.dwa_config.dt * len(path)), len(path) - 1)
            path_point = path[path_point_index]
            
            for obs in predicted_obstacles:
                dist = math.hypot(path_point[0] - obs[0], path_point[1] - obs[1])
                if dist <= (self.robot_radius + obs[2]) * 1.5:  # Add safety margin
                    return False
        return True

    def get_current_obstacles(self, dynamic_obstacles, time):
        """Get current positions of all obstacles."""
        current = self.static_obstacles.copy()
        current.extend(self.predict_obstacle_positions(dynamic_obstacles, time))
        return current

    def move_to_goal(self, start, goal, dynamic_obstacles):
        print(f"Starting navigation from {start} to {goal}")
        x = np.array(list(start) + [math.pi / 8.0, 0.0, 0.0])
        trajectory = np.array([x])
        current_path = None
        last_replan_time = 0
        simulation_time = 0
        in_dwa_mode = False

        while True:
            dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
            
            if simulation_time % 5.0 == 0:  # Print every 5 seconds
                print(f"Time: {simulation_time:.1f}s, Distance to goal: {dist_to_goal:.2f}, Mode: {'DWA' if in_dwa_mode else 'A*'}")

            current_pos = x[:2]
            current_obstacles = self.get_current_obstacles(dynamic_obstacles, simulation_time)
            
            # Less frequent replanning checks
            need_replan = (simulation_time - last_replan_time >= self.replan_interval or 
                        current_path is None)  # Removed path safety check to reduce computation

            if need_replan and not in_dwa_mode:
                print(f"Replanning at time {simulation_time}")
                new_path = self.custom_astar.plan_path(tuple(current_pos), goal, current_obstacles)
                if new_path:
                    current_path = new_path
                    last_replan_time = simulation_time
                else:
                    in_dwa_mode = True
            # Convert dynamic obstacles for DWA
            dwa_obstacles = []
            for obs in current_obstacles:
                dwa_obstacles.append([obs[0], obs[1]])
            dwa_obstacles = np.array(dwa_obstacles)

            # Determine if we should use DWA
            use_dwa = in_dwa_mode
            if not use_dwa and current_path:
                for obs in current_obstacles:
                    dist = math.hypot(current_pos[0] - obs[0], current_pos[1] - obs[1])
                    if dist < self.dwa_activation_distance:
                        print(f"Obstacle too close ({dist}), switching to DWA")
                        use_dwa = True
                        break

            # Execute movement
            try:
                if use_dwa:
                    print("Using DWA control")
                    u, _ = dwa_control(x, self.dwa_config, goal, dwa_obstacles)
                    if simulation_time - last_replan_time >= self.replan_interval:
                        test_path = self.custom_astar.plan_path(tuple(current_pos), goal, current_obstacles)
                        if test_path and self.check_path_safety(test_path, dynamic_obstacles, simulation_time):
                            current_path = test_path
                            in_dwa_mode = False
                            print("Switching back to A* mode")
                else:
                    print("Following A* path")
                    target_idx = min(range(len(current_path)), key=lambda i: 
                        math.hypot(current_path[i][0] - current_pos[0],
                                current_path[i][1] - current_pos[1]))
                    target_idx = min(target_idx + 1, len(current_path) - 1)
                    target = current_path[target_idx]
                    
                    angle = math.atan2(target[1] - x[1], target[0] - x[0])
                    speed = min(self.dwa_config.max_speed, 
                            math.hypot(target[0] - x[0], target[1] - x[1]))
                    u = np.array([speed, angle - x[2]])
            except Exception as e:
                print(f"Error during movement execution: {e}")
                break

            # Update state
            x = motion(x, u, self.dwa_config.dt)
            trajectory = np.vstack((trajectory, x))
            simulation_time += self.dwa_config.dt

            # Check if goal is reached
            if dist_to_goal <= self.resolution:
                print("Goal reached!")
                break

            # Add timeout condition
            if simulation_time > 140.0:  # 30 seconds timeout
                print("Timeout reached")
                break

            # Print state at each iteration
            if len(trajectory) % 50 == 0:  # Print every 50 iterations
                print(f"Current position: {current_pos}, Iterations: {len(trajectory)}")

        return trajectory
      
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