import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from scipy.spatial import distance
import logging
import threading
import csv
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary classes and functions from your existing scripts
from customastar import CustomAStar, CustomAStarAgent, Config as CustomConfig
from dynamicwindow2D import dwa_control, motion, Config as DWAConfig, RobotType
from worksSomewhat2 import AStarDWAAgent

# Disable show_animation globally
import sys
sys.modules['customastar'].show_animation = False
sys.modules['dynamicwindow2D'].show_animation = False
sys.modules['worksSomewhat2'].show_animation = False

class PerformanceMetrics:
    def __init__(self):
        self.execution_times = []
        self.path_length = 0
        self.trajectory = None
        self.success = False

def run_with_timeout(func, args, timeout):
    result = [None]
    def worker():
        try:
            result[0] = func(*args)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            result[0] = None
    
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        logging.warning(f"{func.__name__} timed out after {timeout} seconds")
        return None
    return result[0]

def run_custom_astar(start, goal, obstacles, speed):
    config = CustomConfig()
    config.max_speed = speed
    agent = CustomAStarAgent(2.0, 1.0)
    agent.config.max_speed = speed
    
    metrics = PerformanceMetrics()
    start_time = time.time()
    
    path = agent.plan_path(start, goal, obstacles)
    if not path:
        logging.warning("CustomA* failed to find a path")
        return metrics
    
    trajectory = agent.move_to_goal(list(start), list(goal), path, obstacles)
    end_time = time.time()
    
    metrics.execution_times = [end_time - start_time]
    metrics.path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
    metrics.trajectory = trajectory
    metrics.success = True
    
    return metrics

def run_dwa(start, goal, obstacles, speed):
    config = DWAConfig()
    config.max_speed = speed
    config.robot_type = RobotType.circle
    
    x = np.array(list(start) + [math.pi / 8.0, 0.0, 0.0])
    trajectory = np.array([x])
    
    metrics = PerformanceMetrics()
    start_time = time.time()
    max_iterations = 1000
    
    for i in range(max_iterations):
        u, _ = dwa_control(x, config, goal, obstacles)
        x = motion(x, u, config.dt)
        trajectory = np.vstack((trajectory, [x]))
        
        if (i + 1) % 10 == 0:  # Record time every 10 iterations
            metrics.execution_times.append(time.time() - start_time)
        
        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= config.robot_radius:
            metrics.success = True
            break
    else:
        logging.warning("DWA reached maximum iterations without reaching the goal")
    
    metrics.path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
    metrics.trajectory = trajectory
    
    return metrics

def run_astar_dwa(start, goal, obstacles, speed):
    agent = AStarDWAAgent(obstacles, 2.0, 1.0)
    agent.dwa_config.max_speed = speed
    
    metrics = PerformanceMetrics()
    start_time = time.time()
    
    path = agent.plan_path(start[0], start[1], goal[0], goal[1])
    if not path or len(path[0]) == 0:
        logging.warning("A* + DWA failed to find a path")
        return metrics
    
    trajectory = agent.move_to_goal(list(start), list(goal), path, [])
    end_time = time.time()
    
    metrics.execution_times = [end_time - start_time]
    metrics.path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
    metrics.trajectory = trajectory
    metrics.success = True
    
    return metrics

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
    
    obstacle_counts = list(results.keys())
    algorithms = ['Custom A*', 'DWA', 'A* + DWA']
    colors = ['r', 'g', 'b']
    
    for i, algorithm in enumerate(algorithms):
        execution_times = []
        path_lengths = []
        for count in obstacle_counts:
            metrics = results[count][i]
            if metrics['success']:
                execution_times.append(metrics['time'])
                path_lengths.append(metrics['path_length'])
            else:
                execution_times.append(float('nan'))
                path_lengths.append(float('nan'))
        
        ax1.plot(obstacle_counts, execution_times, f'-{colors[i]}o', label=algorithm)
        ax2.plot(obstacle_counts, path_lengths, f'-{colors[i]}o', label=algorithm)
        
        for j, (time, length) in enumerate(zip(execution_times, path_lengths)):
            if math.isnan(time):
                ax1.text(obstacle_counts[j], ax1.get_ylim()[1], 'Failed', 
                         ha='center', va='bottom', color=colors[i], rotation=90)
                ax2.text(obstacle_counts[j], ax2.get_ylim()[1], 'Failed', 
                         ha='center', va='bottom', color=colors[i], rotation=90)
    
    ax1.set_xlabel('Number of Obstacles')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Algorithm Execution Time Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Number of Obstacles')
    ax2.set_ylabel('Path Length')
    ax2.set_title('Algorithm Path Length Comparison')
    ax2.legend()
    ax2.grid(True)
    
    fig.tight_layout()
    fig.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_random_obstacles(num_obstacles, start, goal, arena_size=50):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            x = random.uniform(0, arena_size)
            y = random.uniform(0, arena_size)
            if (distance.euclidean((x, y), start) > 3 and 
                distance.euclidean((x, y), goal) > 3):
                obstacles.append((x, y, 1.0))  # 1.0 is the obstacle radius
                break
    return obstacles

def run_trial(start, goal, obstacles, speed):
    custom_astar_metrics = run_with_timeout(run_custom_astar, (start, goal, obstacles, speed), 60)
    dwa_metrics = run_with_timeout(run_dwa, (start, goal, np.array(obstacles)[:, :2], speed), 60)
    astar_dwa_metrics = run_with_timeout(run_astar_dwa, (start, goal, obstacles, speed), 60)
    
    return custom_astar_metrics, dwa_metrics, astar_dwa_metrics

def average_metrics(metrics_list):
    success_count = sum(1 for m in metrics_list if m and m.success)
    if success_count == 0:
        return {'success': False, 'time': 0, 'path_length': 0}
    
    avg_time = np.mean([m.execution_times[-1] for m in metrics_list if m and m.success])
    avg_path_length = np.mean([m.path_length for m in metrics_list if m and m.success])
    
    return {'success': success_count > len(metrics_list) // 2, 'time': avg_time, 'path_length': avg_path_length}

def main():
    start = (5.0, 5.0)
    goal = (45.0, 45.0)
    speed = 1.0
    
    results = {}
    
    for obstacle_count in range(5, 10, 5):
        print(f"Running simulations with {obstacle_count} obstacles")
        
        custom_astar_results = []
        dwa_results = []
        astar_dwa_results = []
        
        for _ in range(10):
            obstacles = generate_random_obstacles(obstacle_count, start, goal)
            trial_results = run_trial(start, goal, obstacles, speed)
            custom_astar_results.append(trial_results[0])
            dwa_results.append(trial_results[1])
            astar_dwa_results.append(trial_results[2])
        
        results[obstacle_count] = [
            average_metrics(custom_astar_results),
            average_metrics(dwa_results),
            average_metrics(astar_dwa_results)
        ]
    
    with open('algorithm_comparison_results_obs.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Obstacles', 'CustomA*_Time', 'CustomA*_PathLength', 'DWA_Time', 'DWA_PathLength', 'AStarDWA_Time', 'AStarDWA_PathLength'])
        
        for obstacle_count, metrics in results.items():
            row = [obstacle_count]
            for alg_metrics in metrics:
                if alg_metrics['success']:
                    row.extend([alg_metrics['time'], alg_metrics['path_length']])
                else:
                    row.extend([0, 0])
            csvwriter.writerow(row)
    
    plot_results(results)

if __name__ == "__main__":
    main()