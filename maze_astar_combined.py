import numpy as np
import csv
import time
from maze import MazeGenerator
from LookingGood import AStarDWAAgent
from customastar import CustomAStarAgent
from dynamicwindow2D import Config, dwa_control, motion

# Disable matplotlib.pyplot
import matplotlib
matplotlib.use('Agg')

def run_simulation_astar_dwa(maze, start, goal, num_runs=10):
    resolution = 1.0
    robot_radius = 0.5
    max_speed = 1.0

    obstacles = MazeGenerator.maze_to_obstacles(maze)
    agent = AStarDWAAgent(obstacles, resolution, robot_radius, max_speed)

    total_path_length = 0
    total_time = 0

    for _ in range(num_runs):
        start_time = time.time()
        trajectory = agent.move_to_goal(list(start), list(goal), [])
        end_time = time.time()

        path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
        total_path_length += path_length
        total_time += end_time - start_time

    avg_path_length = total_path_length / num_runs
    avg_time = total_time / num_runs

    return avg_path_length, avg_time

def run_simulation_custom_astar(maze, start, goal, num_runs=10):
    grid_size = 1.0
    robot_radius = 0.5

    obstacles = MazeGenerator.maze_to_obstacles(maze)
    agent = CustomAStarAgent(grid_size, robot_radius)

    total_path_length = 0
    total_time = 0

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

    avg_path_length = total_path_length / num_runs
    avg_time = total_time / num_runs

    return avg_path_length, avg_time

def run_simulation_dwa(maze, start, goal, num_runs=10):
    config = Config()
    config.robot_radius = 0.5
    config.max_speed = 1.0

    obstacles = MazeGenerator.maze_to_obstacles(maze)
    ob = np.array([[obs[0], obs[1]] for obs in obstacles])

    total_path_length = 0
    total_time = 0

    for _ in range(num_runs):
        start_time = time.time()
        x = np.array(start + [0.0, 0.0, 0.0])
        trajectory = np.array(x)

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

    avg_path_length = total_path_length / num_runs
    avg_time = total_time / num_runs

    return avg_path_length, avg_time

def main():
    maze_gen = MazeGenerator(50, 50)
    algorithms = [
        ("AStarDWA", run_simulation_astar_dwa),
        ("CustomAStar", run_simulation_custom_astar),
        ("DynamicWindow2D", run_simulation_dwa)
    ]

    for algo_name, algo_func in algorithms:
        results = []
        for difficulty in range(1, 11):
            maze = maze_gen.get_maze(difficulty)
            start = maze_gen.start
            goal = maze_gen.goal

            print(f"Running {algo_name} simulation for maze difficulty {difficulty}...")
            avg_path_length, avg_time = algo_func(maze, start, goal)
            
            results.append({
                'Difficulty': difficulty,
                'Avg Path Length': avg_path_length,
                'Avg Time': avg_time
            })

            print(f"Maze {difficulty}: Avg Path Length = {avg_path_length:.2f}, Avg Time = {avg_time:.2f}s")

        with open(f'{algo_name.lower()}_maze_simulation_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Difficulty', 'Avg Path Length', 'Avg Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"{algo_name} simulation complete. Results saved to {algo_name.lower()}_maze_simulation_results.csv")

if __name__ == "__main__":
    main()