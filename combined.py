
import numpy as np
import csv
import time
import os
from updated_maze_show_all_difficulties import MazeGenerator
from astar import CustomAStar
from dynamicwindow2d import dwa_control, Config, motion
from LookingGoodv2 import AStarPlanner
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt_lib

if not os.path.exists("/mnt/data/trajectories"):
    os.makedirs("/mnt/data/trajectories")

def save_to_csv(results, filename):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def save_maze_image_with_trajectory(maze, start, goal, trajectory, filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(maze, cmap='binary', origin='lower')
    ax.plot(start[1], start[0], 'go', markersize=8, label="Start")
    ax.plot(goal[1], goal[0], 'ro', markersize=8, label="Goal")

    if len(trajectory) > 0:
        ax.plot([pos[1] for pos in trajectory], [pos[0] for pos in trajectory], 'b-', linewidth=2, alpha=0.7, label="Trajectory")

    ax.legend()
    ax.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_algorithm(algorithm_name, start, goal, obstacles, maze, config=None, run_id=0, difficulty=0):
    result = {}
    trajectory = []
    start_time = time.time()

    if algorithm_name == "CustomAStar":
        custom_astar = CustomAStar(grid_size=0.5, robot_radius=config.robot_radius)
        trajectory = custom_astar.plan(start, goal, obstacles)
    elif algorithm_name == "LookingGood":
        astar_planner = AStarPlanner([], [], resolution=0.5, robot_radius=config.robot_radius)
        ox, oy, radius_list = zip(*obstacles)
        astar_planner.calc_obstacle_map(ox, oy, radius_list)
        trajectory = astar_planner.planning(start[0], start[1], goal[0], goal[1])
    elif algorithm_name == "DWA":
        x = np.array([start[0], start[1], 0.0, 0.0, 0.0])  # [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        while True:
            u, traj = dwa_control(x, config, goal, obstacles)
            x = motion(x, u, config.dt)
            trajectory.append([x[0], x[1]])
            if np.linalg.norm([x[0] - goal[0], x[1] - goal[1]]) <= config.robot_radius:
                break

    result['algorithm'] = algorithm_name
    result['start'] = start
    result['goal'] = goal
    result['time'] = time.time() - start_time
    result['trajectory'] = trajectory

    # Save the trajectory as a PNG file
    trajectory_image_filename = f"/mnt/data/trajectories/{algorithm_name}_difficulty_{difficulty}_run_{run_id}.png"
    save_maze_image_with_trajectory(maze, start, goal, trajectory, trajectory_image_filename)
    result['trajectory_image'] = trajectory_image_filename

    return result

# Initialize the maze generator
maze_gen = MazeGenerator(width=50, height=50)
config = Config()  # for Dynamic Window Approach (DWA)

# Run through each maze difficulty and apply the algorithms
results = []
for difficulty in range(1, 11):
    for i in range(10):  # Run each difficulty 10 times
        maze = maze_gen.generate_maze(difficulty * 2)
        start = maze_gen.start
        goal = maze_gen.goal
        obstacles = maze

        # Run Custom A* Algorithm
        result_astar = run_algorithm("CustomAStar", start, goal, obstacles, maze, config, run_id=i, difficulty=difficulty)
        results.append(result_astar)

        # Run LookingGood Algorithm
        result_lookinggood = run_algorithm("LookingGood", start, goal, obstacles, maze, config, run_id=i, difficulty=difficulty)
        results.append(result_lookinggood)

        # Run Dynamic Window Approach (DWA)
        result_dwa = run_algorithm("DWA", start, goal, obstacles, maze, config, run_id=i, difficulty=difficulty)
        results.append(result_dwa)

# Save all results to CSV
save_to_csv(results, "/mnt/data/maze_algorithms_results_with_images.csv")

print("Finished running all algorithms and saving results and images.")
