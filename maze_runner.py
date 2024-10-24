import numpy as np
import csv
import time
from enum import Enum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LookingGood import AStarDWAAgent
from customastar import CustomAStarAgent
from dynamicwindow2D import Config, dwa_control, motion

class RobotType(Enum):
    circle = 0
    rectangle = 1

class MazeGenerator:
    def __init__(self, width, height, start=(4, 4), goal=(46, 46)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.mazes = [self.generate_maze(difficulty * 2) for difficulty in range(1, 11)]

    def generate_maze(self, obstacle_count):
        maze = np.zeros((self.height, self.width))
        obstacles = []

        # Add boundary walls and convert them to obstacles
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
        
        # Add wall obstacles (top and bottom walls)
        for x in range(self.width):
            obstacles.append((x, 0, 0.5))  # Top wall
            obstacles.append((x, self.height - 1, 0.5))  # Bottom wall
        
        # Add wall obstacles (left and right walls)
        for y in range(self.height):
            obstacles.append((0, y, 0.5))  # Left wall
            obstacles.append((self.width - 1, y, 0.5))  # Right wall

        # Place random circular obstacles
        added_obstacles = 0
        while added_obstacles < obstacle_count:
            i = np.random.randint(1, self.height - 1)
            j = np.random.randint(1, self.width - 1)
            if maze[i, j] == 0 and (j, i) != self.start and (j, i) != self.goal:
                radius = 0.5
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
        return obstacles

    def visualize_maze(self, maze, obstacles, ax):
        ax.imshow(maze, cmap='binary', origin='lower')
        ax.plot(self.start[0], self.start[1], 'go', markersize=8, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Goal')
        for (x, y, r) in obstacles:
            circle = plt.Circle((x, y), r, color='blue', fill=True, alpha=0.5)
            ax.add_artist(circle)
        ax.set_xlim(-1, self.width)
        ax.set_ylim(-1, self.height)
        ax.set_aspect('equal')
        ax.axis('off')

def save_maze_image_with_trajectory(maze, obstacles, start, goal, trajectory, filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(maze, cmap='binary', origin='lower')
    ax.plot(start[1], start[0], 'go', markersize=8)
    ax.plot(goal[1], goal[0], 'ro', markersize=8)
    
    for (x, y, r) in obstacles:
        circle = plt.Circle((x, y), r, color='blue', fill=True, alpha=0.5)
        ax.add_artist(circle)

    ax.plot(trajectory[:, 1], trajectory[:, 0], 'b-', linewidth=2, alpha=0.7)
    
    ax.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_all_mazes(maze_gen, filename):
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Generated Mazes with Increasing Difficulty', fontsize=16)

    for i in range(10):
        maze, obstacles = maze_gen.get_maze(i + 1)
        row = i // 5
        col = i % 5
        maze_gen.visualize_maze(maze, obstacles, axs[row, col])
        axs[row, col].set_title(f'Maze {i+1}')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_simulation_astar_dwa(maze, obstacles, start, goal, num_runs=10, visualize=False):
    resolution = 1.0
    robot_radius = 2
    max_speed = 1.0
    map_height, map_width = maze.shape

    agent = AStarDWAAgent(obstacles, resolution, robot_radius, max_speed, map_width, map_height)

    total_path_length = 0
    total_time = 0
    best_trajectory = None
    best_path_length = float('inf')

    for _ in range(num_runs):
        start_time = time.time()
        trajectory = agent.move_to_goal(list(start), list(goal), [])
        end_time = time.time()

        if len(trajectory) > 1:
            path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
            total_path_length += path_length
            total_time += end_time - start_time

            if path_length < best_path_length:
                best_path_length = path_length
                best_trajectory = trajectory
        else:
            print("No valid trajectory found in this run.")

    if total_path_length > 0:
        avg_path_length = total_path_length / num_runs
        avg_time = total_time / num_runs
    else:
        avg_path_length = 0
        avg_time = 0

    if best_trajectory is None:
        best_trajectory = np.array([start])  

    return avg_path_length, avg_time, best_trajectory

def run_simulation_custom_astar(maze, obstacles, start, goal, num_runs=10, visualize=True):
    grid_size = 1.0
    robot_radius = 1.5
    map_height, map_width = maze.shape

    agent = CustomAStarAgent(grid_size, robot_radius, map_width, map_height)

    total_path_length = 0
    total_time = 0
    best_trajectory = None
    best_path_length = float('inf')

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        path = agent.plan_path(start, goal, obstacles)
        
        if path:
            start_time = time.time()
            # Always simulate the path traversal, visualize if requested
            trajectory = agent.move_to_goal(start, goal, path, obstacles, visualize=visualize)
            end_time = time.time()
            
            path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
            total_path_length += path_length
            total_time += end_time - start_time

            if path_length < best_path_length:
                best_path_length = path_length
                best_trajectory = trajectory
        else:
            print("No path found!")
            trajectory = np.array([start])

    avg_path_length = total_path_length / num_runs
    avg_time = total_time / num_runs

    return avg_path_length, avg_time, best_trajectory


def run_simulation_dwa(maze, obstacles, start, goal, num_runs=10,visualize=False):
    config = Config()
    config.robot_radius = 1.5
    config.max_speed = 1.0

    ob = np.array([[obs[0], obs[1]] for obs in obstacles])

    total_path_length = 0
    total_time = 0
    best_trajectory = None
    best_path_length = float('inf')

    timeout_limit = 90  # timeout

    for run_index in range(num_runs):
        print(f"\nStarting run {run_index + 1}/{num_runs}")
        
        start_time = time.time()
        x = np.array(list(start) + [0.0, 0.0, 0.0])
        trajectory = np.array([x])

        while True:
            u, _ = dwa_control(x, config, goal, ob)
            x = motion(x, u, config.dt)
            trajectory = np.vstack((trajectory, x))

            dist_to_goal = np.hypot(x[0] - goal[0], x[1] - goal[1])
            if dist_to_goal <= config.robot_radius:
                break

            if time.time() - start_time > timeout_limit:
                print(f"Run {run_index + 1} timed out")
                break

        end_time = time.time()

        path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
        total_path_length += path_length
        total_time += end_time - start_time

        if path_length < best_path_length:
            best_path_length = path_length
            best_trajectory = trajectory

    avg_path_length = total_path_length / num_runs
    avg_time = total_time / num_runs

    return avg_path_length, avg_time, best_trajectory


def main():
    maze_gen = MazeGenerator(50, 50)
    algorithms = [
        ("CustomAStar", run_simulation_custom_astar),
        ("AstarDWA",run_simulation_astar_dwa),
        ("DWA",run_simulation_dwa)
    ]

    for algo_name, algo_func in algorithms:
        results = []
        for difficulty in range(1, 11):
            maze, obstacles = maze_gen.get_maze(difficulty)
            start = maze_gen.start
            goal = maze_gen.goal

            print(f"Running {algo_name} simulation for maze difficulty {difficulty}...")
            avg_path_length, avg_time, best_trajectory = algo_func(maze, obstacles, start, goal)
            
            results.append({
                'Difficulty': difficulty,
                'Avg Path Length': avg_path_length,
                'Avg Time': avg_time
            })

            print(f"Maze {difficulty}: Avg Path Length = {avg_path_length:.2f}, Avg Time = {avg_time:.2f}s")

            
            save_maze_image_with_trajectory(maze, obstacles, start, goal, best_trajectory, f'{algo_name.lower()}_maze_{difficulty}.png')

        with open(f'{algo_name.lower()}_maze_simulation_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Difficulty', 'Avg Path Length', 'Avg Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"{algo_name} simulation complete. Results saved to {algo_name.lower()}_maze_simulation_results.csv")

    save_all_mazes(maze_gen, 'all_generated_mazes.png')

if __name__ == "__main__":
    main()