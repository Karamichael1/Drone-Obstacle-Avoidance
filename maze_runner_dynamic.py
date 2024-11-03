import numpy as np
import csv
import time
import math
from enum import Enum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LookingGoodv2 import AStarDWAAgent
from customastar import CustomAStarAgent
from dynamicwindow2D import Config, dwa_control, motion

class MazeGenerator:
    def __init__(self, width, height, start=(4, 4), goal=(44, 44)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.base_dynamic_obstacles = self.generate_base_dynamic_obstacles()
        self.mazes = [self.generate_maze(level) for level in range(1, 11)]
    
    def generate_base_dynamic_obstacles(self):
        # Define colors for dynamic obstacles
        colors = [
            # Original 10 colors
            'red', 'orange', 'purple', 'green', 'blue', 'brown', 'pink', 'gray', 'cyan', 'magenta',
            # Additional 10 colors
            'darkred', 'chocolate', 'indigo', 'darkgreen', 'navy', 'maroon', 'hotpink', 'dimgray', 'teal', 'mediumvioletred'
        ]
        # Create template for dynamic obstacles with different patterns
        templates = [
            # Original obstacles
            {'x': 20, 'y': 20, 'vx':0.05, 'vy': 0.05, 'radius': 0.5},
            {'x': 20, 'y': 25, 'vx': 0.05, 'vy': 0.0, 'radius': 0.5},
            {'x': 27, 'y': 15, 'vx': 0.0, 'vy': 0.05, 'radius': 0.5},
            {'x': 30, 'y': 15, 'vx': -0.05, 'vy': -0.05, 'radius': 0.5},
            {'x': 30, 'y': 30, 'vx': 0.05, 'vy':-0.05, 'radius': 0.5},
            {'x': 5, 'y': 25, 'vx': 0.05, 'vy':0.05, 'radius': 0.5},
            {'x': 25, 'y': 25, 'vx': 0.05, 'vy': 0.05, 'radius':0.5},
            {'x': 40, 'y': 5, 'vx': 0.0, 'vy': 0.05, 'radius': 0.5},
            {'x': 15, 'y': 35, 'vx': -0.05, 'vy': 0.05, 'radius': 0.5},
            {'x': 20, 'y': 30, 'vx':0.05, 'vy':0.05, 'radius': 0.5},
            
            # Additional obstacles with 0.05 speed
            {'x': 35, 'y': 35, 'vx': 0.05, 'vy': 0.0, 'radius': 0.5},  # Horizontal movement
            {'x': 10, 'y': 15, 'vx': 0.0, 'vy': 0.05, 'radius': 0.5},  # Vertical movement
            {'x': 40, 'y': 20, 'vx': 0.05, 'vy': 0.05, 'radius': 0.5}, # Diagonal movement
            {'x': 15, 'y': 40, 'vx': -0.05, 'vy': 0.05, 'radius': 0.5}, # Diagonal movement
            {'x': 35, 'y': 10, 'vx': -0.05, 'vy': 0.0, 'radius': 0.5},  # Horizontal movement
            {'x': 8, 'y': 35, 'vx': 0.05, 'vy': -0.05, 'radius': 0.5},  # Diagonal movement
            {'x': 42, 'y': 42, 'vx': -0.05, 'vy': -0.05, 'radius': 0.5}, # Diagonal movement
            {'x': 25, 'y': 42, 'vx': 0.0, 'vy': -0.05, 'radius': 0.5},   # Vertical movement
            {'x': 12, 'y': 12, 'vx': 0.05, 'vy': 0.05, 'radius': 0.5},   # Diagonal movement
            {'x': 38, 'y': 25, 'vx': -0.05, 'vy': 0.0, 'radius': 0.5},   # Horizontal movement
        ]
        
        # Additional variations of movement patterns
        movement_variations = [
            # For original obstacles
            {'period': 50, 'type': 'reverse'},
            {'period': 30, 'type': 'reverse'},
            {'period': 40, 'type': 'reverse'},
            {'period': 25, 'type': 'reverse'},
            {'period': 35, 'type': 'reverse'},
            {'period': 45, 'type': 'reverse'},
            {'period': 20, 'type': 'reverse'},
            {'period': 55, 'type': 'reverse'},
            {'period': 60, 'type': 'reverse'},
            {'period': 15, 'type': 'reverse'},
            
            # For new obstacles
            {'period': 40, 'type': 'reverse'},
            {'period': 35, 'type': 'reverse'},
            {'period': 45, 'type': 'reverse'},
            {'period': 30, 'type': 'reverse'},
            {'period': 50, 'type': 'reverse'},
            {'period': 25, 'type': 'reverse'},
            {'period': 55, 'type': 'reverse'},
            {'period': 20, 'type': 'reverse'},
            {'period': 45, 'type': 'reverse'},
            {'period': 35, 'type': 'reverse'},
        ]
        
        # Assign colors and movement variations to templates
        for template, color, variation in zip(templates, colors, movement_variations):
            template['color'] = color
            template['movement_variation'] = variation
            template['initial_vx'] = template['vx']  # Store initial velocities
            template['initial_vy'] = template['vy']
            template['time_since_variation'] = 0
        
        return templates
    def update_dynamic_obstacles(self, obstacles, time):
        """Update dynamic obstacles based on their movement variations"""
        updated_obstacles = []
        
        for obs in obstacles:
            new_obs = obs.copy()
            
            # Handle movement variations
            variation = obs['movement_variation']
            if variation['type'] == 'reverse':
                period = variation['period']
                time_in_cycle = time % period
                
                # Reverse direction at half period
                if time_in_cycle < period / 2:
                    new_obs['vx'] = obs['initial_vx']
                    new_obs['vy'] = obs['initial_vy']
                else:
                    new_obs['vx'] = -obs['initial_vx']
                    new_obs['vy'] = -obs['initial_vy']
                
                # Update position based on current velocity
                new_obs['x'] = obs['x'] + new_obs['vx'] * time
                new_obs['y'] = obs['y'] + new_obs['vy'] * time
                
                # Boundary checking and wrapping
                if new_obs['x'] < 0 or new_obs['x'] > self.width:
                    new_obs['x'] = obs['x']  # Reset position
                    new_obs['vx'] = -new_obs['vx']  # Reverse direction
                if new_obs['y'] < 0 or new_obs['y'] > self.height:
                    new_obs['y'] = obs['y']  # Reset position
                    new_obs['vy'] = -new_obs['vy']  # Reverse direction
            
            updated_obstacles.append(new_obs)
        
        return updated_obstacles

    def generate_maze(self, level):
        # Initialize empty maze with only boundaries
        maze = np.zeros((self.height, self.width))
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
        
        # No static obstacles
        static_obstacles = []
        
        # Get dynamic obstacles for this level (2 obstacles per level)
        obstacles_per_level = 2
        num_obstacles = level * obstacles_per_level
        dynamic_obstacles = self.base_dynamic_obstacles[:num_obstacles]
        
        # Ensure start and goal are empty
        maze[self.start[1], self.start[0]] = 0
        maze[self.goal[1], self.goal[0]] = 0
        
        return maze, static_obstacles, dynamic_obstacles
    
    def get_maze(self, level):
        if 1 <= level <= 10:
            return self.mazes[level - 1]
        else:
            raise ValueError("Level must be between 1 and 10")
def save_all_mazes(maze_gen, filename):
    simulation_time = 15  # Time window for showing obstacle movement
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Maze Levels with Increasing Dynamic Obstacles', fontsize=16)

    for i in range(10):
        maze, _, dynamic_obstacles = maze_gen.get_maze(i + 1)
        row = i // 5
        col = i % 5
        
        # Plot empty maze with just boundaries
        axs[row, col].imshow(maze, cmap='binary', origin='lower')
        
        # Plot start and goal
        axs[row, col].plot(maze_gen.start[1], maze_gen.start[0], 'go', 
                          markersize=12, label='Start', zorder=5)
        axs[row, col].plot(maze_gen.goal[1], maze_gen.goal[0], 'ro', 
                          markersize=12, label='Goal', zorder=5)
        
        # Plot dynamic obstacles with movement trails
        for obs in dynamic_obstacles:
            # Draw movement trail
            start_x = obs['x']
            start_y = obs['y']
            
            # Calculate end position considering movement variation
            variation = obs['movement_variation']
            if variation['type'] == 'reverse':
                # For reverse movement, show full cycle
                period = variation['period']
                points_forward = np.linspace(0, period/2, 25)
                points_backward = np.linspace(period/2, period, 25)
                
                # Forward movement
                x_forward = [start_x + obs['vx'] * t for t in points_forward]
                y_forward = [start_y + obs['vy'] * t for t in points_forward]
                
                # Backward movement
                x_backward = [x_forward[-1] - obs['vx'] * t for t in points_backward]
                y_backward = [y_forward[-1] - obs['vy'] * t for t in points_backward]
                
                # Combine points
                x_points = np.concatenate([x_forward, x_backward])
                y_points = np.concatenate([y_forward, y_backward])
            else:
                # Standard linear movement
                end_x = start_x + obs['vx'] * simulation_time
                end_y = start_y + obs['vy'] * simulation_time
                x_points = np.linspace(start_x, end_x, 50)
                y_points = np.linspace(start_y, end_y, 50)
            
            # Draw trail with gradient
            alphas = np.linspace(0.1, 0.7, len(x_points))
            for j in range(len(x_points) - 1):
                axs[row, col].plot([x_points[j], x_points[j+1]], 
                                 [y_points[j], y_points[j+1]], 
                                 '-', color=obs['color'], 
                                 alpha=alphas[j], linewidth=1)
            
            # Draw obstacle positions at intervals
            num_positions = 5
            time_points = np.linspace(0, simulation_time, num_positions)
            for t in time_points:
                current_x = obs['x'] + obs['vx'] * t
                current_y = obs['y'] + obs['vy'] * t
                alpha = 0.6 if t in [0, simulation_time] else 0.3
                circle = plt.Circle((current_x, current_y), obs['radius'],
                                  color=obs['color'], fill=True, alpha=alpha,
                                  zorder=4)
                axs[row, col].add_artist(circle)
        
        # Set plot properties
        axs[row, col].set_title(f'Level {i+1}: {len(dynamic_obstacles)} Dynamic Obstacle{"s" if len(dynamic_obstacles) > 1 else ""}',
                               fontsize=12)
        axs[row, col].set_xlim(-1, maze_gen.width)
        axs[row, col].set_ylim(-1, maze_gen.height)
        axs[row, col].set_aspect('equal')
        
        # Only add legend to first plot
        if i == 0:
            axs[row, col].legend(loc='upper left', fontsize=10)
        
        # Remove axis ticks
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
def convert_walls_to_obstacles(maze):
    wall_obstacles = []
    height, width = maze.shape
    wall_radius = 0.5
    
    # Only convert boundary walls
    for y in range(height):
        for x in range(width):
            if maze[y, x] == 1:
                wall_obstacles.append((x, y, wall_radius))
    return wall_obstacles

def convert_dynamic_to_static(dynamic_obstacles, time):
    static_format = []
    for obs in dynamic_obstacles:
        x = obs['x'] + obs['vx'] * time
        y = obs['y'] + obs['vy'] * time
        static_format.append((x, y, obs['radius']))
    return static_format

def check_collision(position, obstacles, dynamic_obstacles, time, maze=None, goal=None):
    robot_radius = 1.0
    
    # Convert position to numpy array if it isn't already
    position = np.array(position)
    
    # Only do goal checking if goal is provided
    if goal is not None:
        goal = np.array(goal)
        dist_to_goal = np.linalg.norm(position - goal)
        if dist_to_goal <= robot_radius * 2:
            return False
    
    if maze is not None:
        grid_x = int(round(position[0]))
        grid_y = int(round(position[1]))
        radius_check = int(np.ceil(robot_radius))
        for dx in range(-radius_check, radius_check + 1):
            for dy in range(-radius_check, radius_check + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                if (0 <= check_x < maze.shape[1] and 0 <= check_y < maze.shape[0]):
                    if maze[check_y, check_x] == 1:
                        distance = np.linalg.norm([position[0] - check_x, position[1] - check_y])
                        if distance <= robot_radius:
                            return True
    
    for obs in obstacles:
        obs_pos = np.array([obs[0], obs[1]])
        distance = np.linalg.norm(position - obs_pos)
        if distance <= (robot_radius + obs[2]):
            return True
    
    for obs in dynamic_obstacles:
        current_x = obs['x'] + obs['vx'] * time
        current_y = obs['y'] + obs['vy'] * time
        obs_pos = np.array([current_x, current_y])
        distance = np.linalg.norm(position - obs_pos)
        if distance <= (robot_radius + obs['radius']):
            return True
    
    return False
def draw_movement_trail(ax, start_x, start_y, end_x, end_y, color, alpha_gradient=True):
    num_points = 50
    x_points = np.linspace(start_x, end_x, num_points)
    y_points = np.linspace(start_y, end_y, num_points)
    
    if alpha_gradient:
        alphas = np.linspace(0.1, 0.7, num_points)
        for i in range(len(x_points) - 1):
            ax.plot([x_points[i], x_points[i+1]], 
                   [y_points[i], y_points[i+1]], 
                   '-', color=color, alpha=alphas[i], linewidth=1)
    else:
        ax.plot(x_points, y_points, '--', color=color, alpha=0.5, linewidth=1)

def save_maze_image_with_trajectory(maze, obstacles, dynamic_obstacles, start, goal, trajectory, filename):
    """
    Save a visualization of the maze with trajectory, matching the style of maze_runner.py
    while adding support for dynamic obstacles.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the base maze
    ax.imshow(maze, cmap='binary', origin='lower')
    
    # Plot start and goal positions - note using [1], [0] order to match maze_runner.py
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal', zorder=5)
    
    # Plot static obstacles
    for (x, y, r) in obstacles:
        circle = plt.Circle((x, y), r, color='blue', fill=True, alpha=0.5)
        ax.add_artist(circle)
    
    # Plot dynamic obstacles with their trails
    simulation_time = 15  # Time window for showing obstacle movement
    for obs in dynamic_obstacles:
        start_x = obs['x']
        start_y = obs['y']
        
        # Calculate positions for obstacle movement visualization
        num_points = 50
        times = np.linspace(0, simulation_time, num_points)
        x_points = start_x + obs['vx'] * times
        y_points = start_y + obs['vy'] * times
        
        # Draw movement trail with gradient
        alphas = np.linspace(0.1, 0.7, len(x_points))
        for i in range(len(x_points) - 1):
            ax.plot([x_points[i], x_points[i+1]], 
                   [y_points[i], y_points[i+1]], 
                   '-', color=obs['color'], 
                   alpha=alphas[i], linewidth=1)
        
        # Show obstacle positions at key intervals
        num_positions = 5
        for i in range(num_positions):
            t = i * simulation_time / (num_positions - 1)
            current_x = obs['x'] + obs['vx'] * t
            current_y = obs['y'] + obs['vy'] * t
            alpha = 0.6 if i in [0, num_positions-1] else 0.3
            circle = plt.Circle((current_x, current_y), obs['radius'],
                              color=obs['color'], fill=True, alpha=alpha)
            ax.add_artist(circle)
    
    # Plot trajectory if available - using the same style as maze_runner.py
    if trajectory is not None and len(trajectory) > 0:
        # Plot the main trajectory line
        ax.plot(trajectory[:, 1], trajectory[:, 0], 'b-', 
               linewidth=2, alpha=0.7, zorder=3)
        
        # Add direction arrows along the trajectory
        num_arrows = min(20, len(trajectory) - 1)
        arrow_indices = np.linspace(0, len(trajectory) - 2, num_arrows, dtype=int)
        
        for i in arrow_indices:
            dx = trajectory[i + 1, 1] - trajectory[i, 1]
            dy = trajectory[i + 1, 0] - trajectory[i, 0]
            ax.arrow(trajectory[i, 1], trajectory[i, 0], 
                    dx/2, dy/2,
                    head_width=0.3, head_length=0.5,
                    fc='blue', ec='blue', alpha=0.7,
                    zorder=4)
    
    # Set plot properties
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-1, maze.shape[1])
    ax.set_ylim(-1, maze.shape[0])
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
def run_simulation_custom_astar(maze, obstacles, start, goal, dynamic_obstacles, num_runs=10, visualize=True):
    grid_size = 1.0
    robot_radius = 1.5
    map_height, map_width = maze.shape
    agent = CustomAStarAgent(grid_size, robot_radius, map_width, map_height)
    
    wall_obstacles = convert_walls_to_obstacles(maze)
    all_obstacles = obstacles + wall_obstacles
    
    total_path_length = 0
    total_time = 0
    successful_runs = 0
    best_trajectory = None
    best_path_length = float('inf')

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        start_time = time.time()
        
        # Initial path planning with current obstacle positions
        initial_dynamic_obs = convert_dynamic_to_static(dynamic_obstacles, 0)
        current_obstacles = all_obstacles + initial_dynamic_obs
        path = agent.plan_path(start, goal, current_obstacles)
        
        if path:
            # Use agent's move_to_goal method to simulate movement
            trajectory = agent.move_to_goal(start, goal, path, current_obstacles, visualize=visualize)
            
            end_time = time.time()
            # Calculate path length using only position components
            positions = trajectory[:, :2]  # Extract x, y coordinates
            path_length = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
            
            total_path_length += path_length
            total_time += end_time - start_time
            successful_runs += 1
            
            if path_length < best_path_length:
                best_path_length = path_length
                best_trajectory = trajectory
        else:
            print(f"Run {run + 1}: No valid path found")

    if successful_runs > 0:
        avg_path_length = total_path_length / successful_runs
        avg_time = total_time / successful_runs
    else:
        avg_path_length = float('inf')
        avg_time = float('inf')
    
    success_rate = successful_runs / num_runs
    print(f"Custom A* Success rate: {success_rate * 100:.2f}%")
    
    return avg_path_length, avg_time, best_trajectory, success_rate

def run_simulation_astar_dwa(maze, obstacles, start, goal, dynamic_obstacles, num_runs=10, visualize=False):
    resolution = 1.0
    robot_radius = 2.0  # Increased from 1.5 to be more conservative
    max_speed = 1.0
    replot_interval = 4.0  # Replot every second to react faster to obstacles
    map_height, map_width = maze.shape
    config = Config()
    
    wall_obstacles = convert_walls_to_obstacles(maze)
    all_static_obstacles = obstacles + wall_obstacles
    
    total_path_length = 0
    total_time = 0
    successful_runs = 0
    best_trajectory = None
    best_path_length = float('inf')

    for run in range(num_runs):
        print(f"\nStarting A*-DWA run {run + 1}/{num_runs}")
        start_time = time.time()
        simulation_time = 0
        last_replot_time = -replot_interval
        trajectory_points = []
        current_pos = list(start)
        
        while True:
            # Update dynamic obstacle positions
            current_dynamic_obs = []
            for obs in dynamic_obstacles:
                variation = obs['movement_variation']
                period = variation['period']
                time_in_cycle = simulation_time % period
                
                if time_in_cycle < period / 2:
                    current_x = obs['x'] + obs['initial_vx'] * time_in_cycle
                    current_y = obs['y'] + obs['initial_vy'] * time_in_cycle
                    current_vx = obs['initial_vx']
                    current_vy = obs['initial_vy']
                else:
                    time_in_reverse = time_in_cycle - period / 2
                    current_x = obs['x'] + obs['initial_vx'] * (period/2) - obs['initial_vx'] * time_in_reverse
                    current_y = obs['y'] + obs['initial_vy'] * (period/2) - obs['initial_vy'] * time_in_reverse
                    current_vx = -obs['initial_vx']
                    current_vy = -obs['initial_vy']
                
                # Inflate obstacle radius for safety
                current_dynamic_obs.append({
                    'x': current_x,
                    'y': current_y,
                    'vx': current_vx,
                    'vy': current_vy,
                    'radius': obs['radius'] * 2.0,  # Double the radius for safety
                    'color': obs['color'],
                    'movement_variation': obs['movement_variation'],
                    'initial_vx': obs['initial_vx'],
                    'initial_vy': obs['initial_vy']
                })
            
            # Check for potential collisions with current position
            current_pos_array = np.array(current_pos)
            is_safe = True
            for obs in current_dynamic_obs:
                dist = np.hypot(current_pos[0] - obs['x'], current_pos[1] - obs['y'])
                if dist < (robot_radius + obs['radius']) * 1.5:  # Add 50% safety margin
                    is_safe = False
                    break
            
            # Force replan if current position is unsafe
            if not is_safe:
                last_replot_time = simulation_time - replot_interval
            
            # Regular replanning check
            if simulation_time - last_replot_time >= replot_interval:
                print(f"Replotting path at time {simulation_time:.2f}s")
                last_replot_time = simulation_time
                
                # Convert obstacles to static format with increased safety margins
                current_static_obs = []
                for obs in current_dynamic_obs:
                    # Project obstacle position forward in time
                    future_x = obs['x'] + obs['vx'] * replot_interval
                    future_y = obs['y'] + obs['vy'] * replot_interval
                    current_static_obs.append((future_x, future_y, obs['radius'] * 2.0))
                
                all_obstacles = all_static_obstacles + current_static_obs
                
                # Create new agent with current obstacle positions
                agent = AStarDWAAgent(all_obstacles, resolution, robot_radius, max_speed, map_width, map_height)
                
                # Plan new path from current position to goal
                path = agent.move_to_goal(current_pos, list(goal), current_dynamic_obs)
                
                if path is None or (isinstance(path, np.ndarray) and len(path) <= 1):
                    print(f"No valid path found at time {simulation_time:.2f}s")
                    # Wait in place for a short time before trying again
                    simulation_time += config.dt
                    continue
                
                # Extract only x,y coordinates and validate path points
                if isinstance(path, np.ndarray):
                    new_points = path[:, :2]
                else:
                    new_points = np.array([[p[0], p[1]] for p in path])
                
                # Validate each point in the path
                valid_points = []
                for point in new_points:
                    is_point_safe = True
                    for obs in current_dynamic_obs:
                        dist = np.hypot(point[0] - obs['x'], point[1] - obs['y'])
                        if dist < (robot_radius + obs['radius']) * 1.5:
                            is_point_safe = False
                            break
                    if is_point_safe:
                        valid_points.append(point)
                
                if not valid_points:
                    print("No safe points found in path")
                    simulation_time += config.dt
                    continue
                
                # Add validated points to trajectory
                if not trajectory_points:
                    trajectory_points.append([current_pos[0], current_pos[1]])
                
                # Only use points up to the next replan time
                points_to_use = min(len(valid_points), int(replot_interval / config.dt) + 1)
                trajectory_points.extend(valid_points[1:points_to_use])
                
                # Update current position
                current_pos = valid_points[points_to_use-1].tolist()
            
            simulation_time += config.dt
            
            # Check if goal reached
            dist_to_goal = np.hypot(current_pos[0] - goal[0], current_pos[1] - goal[1])
            if dist_to_goal <= robot_radius:
                print(f"Goal reached in run {run + 1}")
                successful_runs += 1
                break
            
            # Check timeout
            if time.time() - start_time > 180:
                print(f"Run {run + 1} timed out")
                break
        
        # Convert trajectory points to numpy array
        trajectory = np.array(trajectory_points)
        
        # Calculate path length and update statistics
        if len(trajectory) > 1:
            path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))
            total_path_length += path_length
            total_time += time.time() - start_time
            
            if path_length < best_path_length:
                best_path_length = path_length
                best_trajectory = trajectory

    if successful_runs > 0:
        avg_path_length = total_path_length / successful_runs
        avg_time = total_time / successful_runs
    else:
        avg_path_length = float('inf')
        avg_time = float('inf')
    
    success_rate = successful_runs / num_runs
    print(f"A*-DWA Success rate: {success_rate * 100:.2f}%")
    
    return avg_path_length, avg_time, best_trajectory, success_rate
def run_simulation_dwa(maze, obstacles, start, goal, dynamic_obstacles, num_runs=10, visualize=False):
    config = Config()
    config.robot_radius = 1.5
    config.max_speed = 1.0
    
    wall_obstacles = convert_walls_to_obstacles(maze)
    all_obstacles = obstacles + wall_obstacles
    static_ob = np.array([[obs[0], obs[1]] for obs in all_obstacles])
    
    total_path_length = 0
    total_time = 0
    successful_runs = 0
    best_trajectory = None
    best_path_length = float('inf')
    timeout_limit = 90

    for run_index in range(num_runs):
        print(f"\nStarting DWA run {run_index + 1}/{num_runs}")
        
        start_time = time.time()
        # Initialize state: x, y, yaw, velocity, angular velocity
        x = np.array([start[0], start[1], 0.0, 0.0, 0.0])
        trajectory = np.array([[x[0], x[1]]])  # Store only x,y coordinates
        simulation_time = 0

        while True:
            # Get current dynamic obstacle positions
            current_obstacles = static_ob.copy()
            for obs in dynamic_obstacles:
                current_x = obs['x'] + obs['vx'] * simulation_time
                current_y = obs['y'] + obs['vy'] * simulation_time
                current_obstacles = np.vstack((current_obstacles, [current_x, current_y]))
            
            # DWA control
            u, _ = dwa_control(x, config, goal, current_obstacles)
            x = motion(x, u, config.dt)
            
            # Append new position to trajectory
            trajectory = np.vstack((trajectory, [x[0], x[1]]))
            
            simulation_time += config.dt
            
            # Check if goal reached
            dist_to_goal = np.hypot(x[0] - goal[0], x[1] - goal[1])
            if dist_to_goal <= config.robot_radius:
                print(f"Goal reached in run {run_index + 1}")
                successful_runs += 1
                break
            
            # Check timeout
            if time.time() - start_time > timeout_limit:
                print(f"Run {run_index + 1} timed out")
                break

        end_time = time.time()
        path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))
        
        total_path_length += path_length
        total_time += end_time - start_time
        
        if path_length < best_path_length:
            best_path_length = path_length
            best_trajectory = trajectory

    avg_path_length = total_path_length / num_runs if successful_runs > 0 else float('inf')
    avg_time = total_time / num_runs if successful_runs > 0 else float('inf')
    success_rate = successful_runs / num_runs
    
    print(f"DWA Success rate: {success_rate * 100:.2f}%")
    
    return avg_path_length, avg_time, best_trajectory, success_rate
def save_maze_image_with_trajectory(maze, obstacles, dynamic_obstacles, start, goal, trajectory, filename):
    simulation_time = 15
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(maze, cmap='binary', origin='lower')
    
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal', zorder=5)
    
    for (x, y, r) in obstacles:
        circle = plt.Circle((x, y), r, color='blue', fill=True, alpha=0.5)
        ax.add_artist(circle)
    
    for obs in dynamic_obstacles:
        start_x = obs['x']
        start_y = obs['y']
        end_x = obs['x'] + obs['vx'] * simulation_time
        end_y = obs['y'] + obs['vy'] * simulation_time
        
        draw_movement_trail(ax, start_x, start_y, end_x, end_y, obs['color'])
        
        num_positions = 5
        for i in range(num_positions):
            t = i * simulation_time / (num_positions - 1)
            current_x = obs['x'] + obs['vx'] * t
            current_y = obs['y'] + obs['vy'] * t
            alpha = 0.3 if i in [0, num_positions-1] else 0.15
            circle = plt.Circle((current_x, current_y), obs['radius'], 
                              color=obs['color'], fill=True, alpha=alpha)
            ax.add_artist(circle)
    
    if trajectory is not None and len(trajectory) > 0:
        points = len(trajectory)
        alphas = np.linspace(0.2, 0.9, points)
        
        for i in range(points - 1):
            ax.plot([trajectory[i][1], trajectory[i+1][1]], 
                   [trajectory[i][0], trajectory[i+1][0]], 
                   'b-', alpha=alphas[i], linewidth=2, zorder=3)
        
        arrow_indices = np.linspace(0, points-1, min(20, points), dtype=int)
        for i in arrow_indices:
            if i < points - 1:
                dx = trajectory[i+1][1] - trajectory[i][1]
                dy = trajectory[i+1][0] - trajectory[i][0]
                plt.arrow(trajectory[i][1], trajectory[i][0], dx/2, dy/2,
                         head_width=0.3, head_length=0.5, fc='blue', ec='blue',
                         alpha=alphas[i], zorder=4)
    
    ax.legend(loc='upper right', fontsize=12)
    plt.title(f'Path Execution', fontsize=14)
    
    ax.set_xlim(-1, maze.shape[1] + 1)
    ax.set_ylim(-1, maze.shape[0] + 1)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    maze_gen = MazeGenerator(50, 50)
    algorithms = [
        # ("CustomAStar", run_simulation_custom_astar),
        ("AstarDWA", run_simulation_astar_dwa),
        ("DWA", run_simulation_dwa)
    ]
    save_all_mazes(maze_gen, 'all_generated_mazes.png')

    for algo_name, algo_func in algorithms:
        results = []
        for level in range(1, 11):
            maze, static_obstacles, dynamic_obstacles = maze_gen.get_maze(level)
            start = maze_gen.start
            goal = maze_gen.goal

            print(f"Running {algo_name} simulation for level {level} with {level} dynamic obstacles...")
            avg_path_length, avg_time, best_trajectory, success_rate = algo_func(
                maze, static_obstacles, start, goal, dynamic_obstacles)
            
            results.append({
                'Level': level,
                'Dynamic Obstacles': level,
                'Avg Path Length': avg_path_length,
                'Avg Time': avg_time,
                'Success Rate': success_rate
            })

            print(f"Level {level}: Avg Path Length = {avg_path_length:.2f}, "
                  f"Avg Time = {avg_time:.2f}s, Success Rate = {success_rate:.2%}")
            
            save_maze_image_with_trajectory(
                maze, static_obstacles, dynamic_obstacles, start, goal, best_trajectory,
                f'{algo_name.lower()}_level_{level}.png')

        with open(f'{algo_name.lower()}_dynamic_simulation_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Level', 'Dynamic Obstacles', 'Avg Path Length', 'Avg Time', 'Success Rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

if __name__ == "__main__":
    main()