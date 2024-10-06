import numpy as np
import csv
import time
from maze import MazeGenerator
from LookingGood import AStarDWAAgent
from customastar import CustomAStarAgent
from dynamicwindow2D import Config, dwa_control, motion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt_lib

def save_maze_image_with_trajectory(maze, start, goal, trajectory, filename):
    fig = plt_lib.figure(figsize=(5, 5))
    plt_lib.imshow(maze, cmap='binary')
    plt_lib.plot(start[1], start[0], 'go', markersize=8)
    plt_lib.plot(goal[1], goal[0], 'ro', markersize=8)
    

    plt_lib.plot(trajectory[:, 1], trajectory[:, 0], 'b-', linewidth=2, alpha=0.7)
    
    plt_lib.axis('off')
    plt_lib.savefig(filename, dpi=300, bbox_inches='tight')
    plt_lib.close(fig)

def save_all_mazes(mazes, starts, goals, filename):
    fig, axs = plt_lib.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Generated Mazes with Increasing Difficulty', fontsize=16)

    for i, (maze, start, goal) in enumerate(zip(mazes, starts, goals)):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(maze, cmap='binary')
        axs[row, col].plot(start[1], start[0], 'go', markersize=8)
        axs[row, col].plot(goal[1], goal[0], 'ro', markersize=8)
        axs[row, col].set_title(f'Maze {i+1}')
        axs[row, col].axis('off')

    plt_lib.tight_layout()
    plt_lib.savefig(filename, dpi=300, bbox_inches='tight')
    plt_lib.close(fig)
    
def run_simulation_astar_dwa(maze, start, goal, num_runs=10):
    resolution = 1.0
    robot_radius = 0.5
    max_speed = 1.0

    obstacles = MazeGenerator.maze_to_obstacles(maze)
    agent = AStarDWAAgent(obstacles, resolution, robot_radius, max_speed)

    total_path_length = 0
    total_time = 0
    best_trajectory = None
    best_path_length = float('inf')

    for _ in range(num_runs):
        start_time = time.time()
        trajectory = agent.move_to_goal(list(start), list(goal), [])
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

def run_simulation_custom_astar(maze, start, goal, num_runs=10):
    grid_size = 1.0
    robot_radius = 0.5

    obstacles = MazeGenerator.maze_to_obstacles(maze)
    agent = CustomAStarAgent(grid_size, robot_radius)

    total_path_length = 0
    total_time = 0
    best_trajectory = None
    best_path_length = float('inf')

    for _ in range(num_runs):
        start_time = time.time()
        path = agent.plan_path(start, goal, obstacles)
        if path:
            trajectory = agent.move_to_goal(list(start), list(goal), path, obstacles)
        else:
            trajectory = np.array([start])
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

def run_simulation_dwa(maze, start, goal, num_runs=10):
    config = Config()
    config.robot_radius = 0.5
    config.max_speed = 1.0

    obstacles = MazeGenerator.maze_to_obstacles(maze)
    ob = np.array([[obs[0], obs[1]] for obs in obstacles])

    total_path_length = 0
    total_time = 0
    best_trajectory = None
    best_path_length = float('inf')

    for _ in range(num_runs):
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
        ("AStarDWA", run_simulation_astar_dwa),
        ("CustomAStar", run_simulation_custom_astar),
        ("DynamicWindow2D", run_simulation_dwa)
    ]

    mazes = []
    starts = []
    goals = []

    for algo_name, algo_func in algorithms:
        results = []
        for difficulty in range(1, 11):
            maze = maze_gen.get_maze(difficulty)
            start = maze_gen.start
            goal = maze_gen.goal

            mazes.append(maze)
            starts.append(start)
            goals.append(goal)

            print(f"Running {algo_name} simulation for maze difficulty {difficulty}...")
            avg_path_length, avg_time, best_trajectory = algo_func(maze, start, goal)
            
            results.append({
                'Difficulty': difficulty,
                'Avg Path Length': avg_path_length,
                'Avg Time': avg_time
            })

            print(f"Maze {difficulty}: Avg Path Length = {avg_path_length:.2f}, Avg Time = {avg_time:.2f}s")

            # Save individual maze image with trajectory
            save_maze_image_with_trajectory(maze, start, goal, best_trajectory, f'{algo_name.lower()}_maze_{difficulty}.png')

        with open(f'{algo_name.lower()}_maze_simulation_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Difficulty', 'Avg Path Length', 'Avg Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"{algo_name} simulation complete. Results saved to {algo_name.lower()}_maze_simulation_results.csv")

    # Save all mazes in a single image
    save_all_mazes(mazes[:10], starts[:10], goals[:10], 'all_generated_mazes.png')

if __name__ == "__main__":
    main()