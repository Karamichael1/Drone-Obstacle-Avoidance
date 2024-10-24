import numpy as np
import csv
import time
from enum import Enum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LookingGoodv2 import AStarDWAAgent
from customastar import CustomAStarAgent
from dynamicwindow2D import Config, dwa_control, motion

class MazeGenerator:
    def __init__(self, width, height, start=(4, 4), goal=(46, 46)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.dynamic_obstacles = self.generate_dynamic_obstacles()
        self.mazes = [self.generate_maze(difficulty * 2) for difficulty in range(1, 11)]
    
    def generate_dynamic_obstacles(self):
        obstacles = [
            {'x': 10, 'y': 10, 'vx': 0.5, 'vy': 0.5, 'radius': 1.0, 'color': 'red'},
            {'x': 10, 'y': 15, 'vx': 0.3, 'vy': 0.3, 'radius': 1.0, 'color': 'orange'},
            {'x': 10, 'y': 25, 'vx': 0.7, 'vy': 0.7, 'radius': 1.0, 'color': 'purple'}
        ]
        return obstacles

    def generate_maze(self, obstacle_count):
        maze = np.zeros((self.height, self.width))
        obstacles = []
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
        
        simulation_time = 15
        for obs in self.dynamic_obstacles:
            start_x = obs['x']
            start_y = obs['y']
            end_x = start_x + obs['vx'] * simulation_time
            end_y = start_y + obs['vy'] * simulation_time
            
            num_points = 20
            for t in np.linspace(0, simulation_time, num_points):
                x = int(start_x + obs['vx'] * t)
                y = int(start_y + obs['vy'] * t)
                if 0 <= x < self.width and 0 <= y < self.height:
                    buffer = 2
                    for dx in range(-buffer, buffer+1):
                        for dy in range(-buffer, buffer+1):
                            if (0 <= x+dx < self.width and 0 <= y+dy < self.height):
                                maze[y+dy, x+dx] = 0

        added_obstacles = 0
        max_attempts = 1000
        attempts = 0
        
        while added_obstacles < obstacle_count and attempts < max_attempts:
            i = np.random.randint(1, self.height - 1)
            j = np.random.randint(1, self.width - 1)
            
            if (maze[i, j] == 0 and (j, i) != self.start and (j, i) != self.goal):
                radius = 0.5
                obstacles.append((j, i, radius))
                maze[i, j] = 1
                added_obstacles += 1
            attempts += 1

        maze[self.start[1], self.start[0]] = 0
        maze[self.goal[1], self.goal[0]] = 0
        return maze, obstacles

    def get_maze(self, difficulty):
        if 1 <= difficulty <= 10:
            return self.mazes[difficulty - 1]
        else:
            raise ValueError("Difficulty must be between 1 and 10")

def convert_walls_to_obstacles(maze):
    wall_obstacles = []
    height, width = maze.shape
    wall_radius = 0.5
    
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

def check_collision(position, obstacles, dynamic_obstacles, time, maze=None):
    robot_radius = 1.0
    
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
                        distance = np.hypot(position[0] - check_x, position[1] - check_y)
                        if distance <= robot_radius:
                            return True
    
    for obs in obstacles:
        distance = np.hypot(position[0] - obs[0], position[1] - obs[1])
        if distance <= (robot_radius + obs[2]):
            return True
    
    for obs in dynamic_obstacles:
        current_x = obs['x'] + obs['vx'] * time
        current_y = obs['y'] + obs['vy'] * time
        distance = np.hypot(position[0] - current_x, position[1] - current_y)
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

def save_maze_image_with_trajectory(maze, obstacles, dynamic_obstacles, start, goal, trajectory, filename, collision_point=None):
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
    
    if collision_point is not None:
        ax.plot(collision_point[1], collision_point[0], 'rx', markersize=15, 
                label='Collision', markeredgewidth=3, zorder=6)
        circle = plt.Circle((collision_point[1], collision_point[0]), 2,
                          color='red', fill=False, linestyle='--', alpha=0.5)
        ax.add_artist(circle)
    
    ax.legend(loc='upper right', fontsize=12)
    if collision_point is not None:
        plt.title(f'Path Execution (Collision Detected)', fontsize=14)
    else:
        plt.title(f'Path Execution (Successful)', fontsize=14)
    
    ax.set_xlim(-1, maze.shape[1] + 1)
    ax.set_ylim(-1, maze.shape[0] + 1)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_simulation_custom_astar(maze, obstacles, start, goal, dynamic_obstacles, num_runs=10, visualize=False):
    grid_size = 1.0
    robot_radius = 1.5
    agent = CustomAStarAgent(grid_size, robot_radius, 50, 50)
    
    wall_obstacles = convert_walls_to_obstacles(maze)
    initial_dynamic_obs = convert_dynamic_to_static(dynamic_obstacles, 0)
    all_obstacles = obstacles + wall_obstacles + initial_dynamic_obs
    
    total_path_length = 0
    total_time = 0
    successful_runs = 0
    best_trajectory = None
    best_path_length = float('inf')
    collision_point = None

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        start_time = time.time()
        
        path = agent.plan_path(start, goal, all_obstacles, visualize=(visualize and run == 0))
        
        if path and len(path) > 0:  # Check if path exists and is not empty
            trajectory = [np.array(start)]  # Start with initial position
            current_pos = np.array(start)
            simulation_time = 0
            collision_occurred = False
            
            # Use enumerate instead of range
            for waypoint in path:
                # Convert waypoint to numpy array
                waypoint = np.array(waypoint)
                
                # Check collision before moving
                if check_collision(current_pos, all_obstacles, dynamic_obstacles, simulation_time, maze):
                    collision_occurred = True
                    collision_point = current_pos
                    break
                
                # Check if dynamic replanning is needed
                if len(trajectory) % 10 == 0:  # Every 10 steps
                    current_dynamic_obs = convert_dynamic_to_static(dynamic_obstacles, simulation_time)
                    need_replan = False
                    
                    # Check distance to dynamic obstacles
                    for obs in current_dynamic_obs:
                        dist_to_obs = np.hypot(current_pos[0] - obs[0], current_pos[1] - obs[1])
                        if dist_to_obs < 5.0:
                            need_replan = True
                            break
                    
                    if need_replan:
                        current_obstacles = obstacles + wall_obstacles + current_dynamic_obs
                        new_path = agent.plan_path(tuple(current_pos), goal, current_obstacles, visualize=False)
                        if new_path and len(new_path) > 1:
                            # Update remaining path
                            path = new_path[1:]  # Skip first point as it's current position
                            break  # Break current loop to start following new path
                
                # Move to new position
                current_pos = waypoint
                trajectory.append(current_pos)
                simulation_time += 0.1
            
            # Add final position if not already added and no collision occurred
            if not collision_occurred and not np.array_equal(current_pos, goal):
                trajectory.append(np.array(goal))
            
            trajectory = np.array(trajectory)
            
            if not collision_occurred:
                end_time = time.time()
                path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
                
                total_path_length += path_length
                total_time += end_time - start_time
                successful_runs += 1
                
                if path_length < best_path_length:
                    best_path_length = path_length
                    best_trajectory = trajectory
                    collision_point = None
            elif collision_occurred and best_trajectory is None:
                best_trajectory = trajectory
        else:
            print("No valid path found for this run")

    if successful_runs > 0:
        avg_path_length = total_path_length / successful_runs
        avg_time = total_time / successful_runs
    else:
        avg_path_length = float('inf')
        avg_time = float('inf')
    
    success_rate = successful_runs / num_runs
    print(f"Success rate: {success_rate * 100:.2f}%")
    
    return avg_path_length, avg_time, best_trajectory, success_rate, collision_point

def run_simulation_astar_dwa(maze, obstacles, start, goal, dynamic_obstacles, num_runs=10, visualize=False):
    resolution = 1.0
    robot_radius = 1.5
    max_speed = 1.0
    map_height, map_width = maze.shape
    
    wall_obstacles = convert_walls_to_obstacles(maze)
    all_obstacles = obstacles + wall_obstacles
    initial_dynamic_obs = convert_dynamic_to_static(dynamic_obstacles, 0)
    all_obstacles.extend(initial_dynamic_obs)
    
    agent = AStarDWAAgent(all_obstacles, resolution, robot_radius, max_speed, map_width, map_height)
    
    total_path_length = 0
    total_time = 0
    successful_runs = 0
    best_trajectory = None
    best_path_length = float('inf')
    collision_point = None

    for run in range(num_runs):
        start_time = time.time()
        trajectory = agent.move_to_goal(list(start), list(goal), dynamic_obstacles)
        
        collision_occurred = False
        for i, pos in enumerate(trajectory):
            if check_collision(pos[:2], all_obstacles, dynamic_obstacles, i * 0.1, maze):
                collision_occurred = True
                collision_point = pos[:2]
                break
        
        if not collision_occurred and len(trajectory) > 1:
            end_time = time.time()
            path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
            
            total_path_length += path_length
            total_time += end_time - start_time
            successful_runs += 1
            
            if path_length < best_path_length:
                best_path_length = path_length
                best_trajectory = trajectory
                collision_point = None
        elif collision_occurred and best_trajectory is None:
            best_trajectory = trajectory

    if successful_runs > 0:
        avg_path_length = total_path_length / successful_runs
        avg_time = total_time / successful_runs
    else:
        avg_path_length = float('inf')
        avg_time = float('inf')
    
    success_rate = successful_runs / num_runs
    print(f"Success rate: {success_rate * 100:.2f}%")
    
    return avg_path_length, avg_time, best_trajectory,  success_rate, collision_point

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
    collision_point = None
    timeout_limit = 90

    for run_index in range(num_runs):
        print(f"\nStarting DWA run {run_index + 1}/{num_runs}")
        
        start_time = time.time()
        x = np.array(list(start) + [0.0, 0.0, 0.0])
        trajectory = np.array([x])
        simulation_time = 0
        collision_occurred = False

        while True:
            current_obstacles = static_ob.copy()
            for obs in dynamic_obstacles:
                current_x = obs['x'] + obs['vx'] * simulation_time
                current_y = obs['y'] + obs['vy'] * simulation_time
                current_obstacles = np.vstack((current_obstacles, [current_x, current_y]))

            u, predicted_trajectory = dwa_control(x, config, goal, current_obstacles)
            x = motion(x, u, config.dt)
            trajectory = np.vstack((trajectory, x))
            
            if check_collision(x[:2], all_obstacles, dynamic_obstacles, simulation_time, maze):
                collision_occurred = True
                collision_point = x[:2]
                print(f"Collision detected in run {run_index + 1}")
                break

            dist_to_goal = np.hypot(x[0] - goal[0], x[1] - goal[1])
            if dist_to_goal <= config.robot_radius:
                print(f"Goal reached in run {run_index + 1}")
                break

            simulation_time += config.dt
            if time.time() - start_time > timeout_limit:
                print(f"Run {run_index + 1} timed out")
                collision_occurred = True
                break

        end_time = time.time()

        if not collision_occurred:
            path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
            total_path_length += path_length
            total_time += end_time - start_time
            successful_runs += 1

            if path_length < best_path_length:
                best_path_length = path_length
                best_trajectory = trajectory
                collision_point = None

    if successful_runs > 0:
        avg_path_length = total_path_length / successful_runs
        avg_time = total_time / successful_runs
    else:
        avg_path_length = float('inf')
        avg_time = float('inf')
        
    success_rate = successful_runs / num_runs
    print(f"DWA Success rate: {success_rate * 100:.2f}%")

    return avg_path_length, avg_time, best_trajectory, success_rate, collision_point

def main():
    maze_gen = MazeGenerator(50, 50)
    algorithms = [
        # ("CustomAStar", run_simulation_custom_astar),
        ("AstarDWA", run_simulation_astar_dwa)
        # ("DWA", run_simulation_dwa)
    ]

    for algo_name, algo_func in algorithms:
        results = []
        for difficulty in range(1, 11):
            maze, obstacles = maze_gen.get_maze(difficulty)
            start = maze_gen.start
            goal = maze_gen.goal
            dynamic_obstacles = maze_gen.dynamic_obstacles

            print(f"Running {algo_name} simulation for maze difficulty {difficulty}...")
            avg_path_length, avg_time, best_trajectory, success_rate, collision_point = algo_func(
                maze, obstacles, start, goal, dynamic_obstacles)
            
            results.append({
                'Difficulty': difficulty,
                'Avg Path Length': avg_path_length,
                'Avg Time': avg_time,
                'Success Rate': success_rate
            })

            print(f"Maze {difficulty}: Avg Path Length = {avg_path_length:.2f}, "
                  f"Avg Time = {avg_time:.2f}s, Success Rate = {success_rate:.2%}")
            
            save_maze_image_with_trajectory(
                maze, obstacles, dynamic_obstacles, start, goal, best_trajectory,
                f'{algo_name.lower()}_maze_{difficulty}.png', collision_point)

        with open(f'{algo_name.lower()}_maze_simulation_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Difficulty', 'Avg Path Length', 'Avg Time', 'Success Rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

if __name__ == "__main__":
    main()