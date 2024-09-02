import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from scipy.spatial import distance
import logging
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary classes and functions from your existing scripts
from customastar import CustomAStar, CustomAStarAgent, Config as CustomConfig
from dynamicwindow2D import dwa_control, motion, Config as DWAConfig, RobotType
from failsafe import AStarDWAAgent

# Disable show_animation globally
import sys
sys.modules['customastar'].show_animation = False
sys.modules['dynamicwindow2D'].show_animation = False
sys.modules['failsafe'].show_animation = False

class PerformanceMetrics:
    def __init__(self):
        self.execution_time = 0
        self.path_length = 0
        self.trajectory = None

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
    
    start_time = time.time()
    path = agent.plan_path(start, goal, obstacles)
    
    if not path:
        logging.warning("CustomA* failed to find a path")
        return None
    
    trajectory = agent.move_to_goal(list(start), list(goal), path, obstacles)
    end_time = time.time()
    
    metrics = PerformanceMetrics()
    metrics.execution_time = end_time - start_time
    metrics.path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
    metrics.trajectory = trajectory
    
    return metrics

def run_dwa(start, goal, obstacles, speed):
    config = DWAConfig()
    config.max_speed = speed
    config.robot_type = RobotType.circle
    
    x = np.array(list(start) + [math.pi / 8.0, 0.0, 0.0])
    trajectory = np.array([x])
    
    start_time = time.time()
    max_iterations = 1000
    for _ in range(max_iterations):
        u, _ = dwa_control(x, config, goal, obstacles)
        x = motion(x, u, config.dt)
        trajectory = np.vstack((trajectory, [x]))
        
        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= config.robot_radius:
            break
    else:
        logging.warning("DWA reached maximum iterations without reaching the goal")
    
    end_time = time.time()
    
    metrics = PerformanceMetrics()
    metrics.execution_time = end_time - start_time
    metrics.path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
    metrics.trajectory = trajectory
    
    return metrics

def run_astar_dwa(start, goal, obstacles, speed):
    agent = AStarDWAAgent(obstacles, 2.0, 1.0)
    agent.dwa_config.max_speed = speed
    
    path = agent.plan_path(start[0], start[1], goal[0], goal[1])
    if not path or len(path[0]) == 0:
        logging.warning("A* + DWA failed to find a path")
        return None
    
    start_time = time.time()
    trajectory = agent.move_to_goal(list(start), list(goal), path, [])
    end_time = time.time()
    
    metrics = PerformanceMetrics()
    metrics.execution_time = end_time - start_time
    metrics.path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
    metrics.trajectory = trajectory
    
    return metrics

def plot_results(results, obstacles, start, goal):
    plt.figure(figsize=(12, 8))
    
    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], fill=True, color='gray')
        plt.gca().add_artist(circle)
    
    # Plot paths
    colors = ['r', 'g', 'b']
    labels = ['Custom A*', 'DWA', 'A* + DWA']
    
    for i, (speed, metrics) in enumerate(results.items()):
        for j, m in enumerate(metrics):
            if m is not None and m.trajectory is not None:
                plt.plot(m.trajectory[:, 0], m.trajectory[:, 1], f'-{colors[j]}', alpha=0.5 + 0.5 * (speed / 5.0), 
                         label=f'{labels[j]} (Speed: {speed})' if i == 0 else "")
    
    # Plot start and goal
    plt.plot(start[0], start[1], 'ko', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'k*', markersize=10, label='Goal')
    
    plt.legend()
    plt.title('Comparison of Path Planning Algorithms at Various Speeds')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.grid(True)
    
    plt.savefig('path_comparison_all_speeds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_time_vs_speed(results):
    plt.figure(figsize=(12, 8))
    
    speeds = list(results.keys())
    algorithms = ['Custom A*', 'DWA', 'A* + DWA']
    colors = ['r', 'g', 'b']
    
    for i, algorithm in enumerate(algorithms):
        times = []
        for speed in speeds:
            if results[speed][i] is not None:
                times.append(results[speed][i].execution_time)
            else:
                times.append(0)  # Use 0 if the algorithm failed
        
        plt.plot(speeds, times, f'-{colors[i]}o', label=algorithm)
    
    plt.xlabel('Speed')
    plt.ylabel('Time to Completion (s)')
    plt.title('Algorithm Performance vs Speed')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('time_vs_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    start = (5.0, 5.0)
    goal = (45.0, 45.0)
    
    obstacles = [
        (10.0, 10.0, 1.0),
        (20.0, 20.0, 1.0),
        (30.0, 26.0, 1.0),
        (40.0, 42.0, 1.0),
        (15.0, 35.0, 1.0),
        (35.0, 15.0, 1.0)
    ]
    
    results = {}
    
    for speed in np.arange(0.5, 10, 0.5):
        print(f"Running simulation with speed: {speed}")
        
        custom_astar_metrics = run_with_timeout(run_custom_astar, (start, goal, obstacles, speed), 60)
        dwa_metrics = run_with_timeout(run_dwa, (start, goal, np.array(obstacles)[:, :2], speed), 60)
        astar_dwa_metrics = run_with_timeout(run_astar_dwa, (start, goal, obstacles, speed), 60)
        
        results[speed] = [custom_astar_metrics, dwa_metrics, astar_dwa_metrics]
        
        for name, metrics in zip(['Custom A*', 'DWA', 'A* + DWA'], results[speed]):
            if metrics:
                print(f"{name} - Time: {metrics.execution_time:.2f}s, Path Length: {metrics.path_length:.2f}")
            else:
                print(f"{name} - Failed to complete within 60 seconds or encountered an error")
    

if __name__ == "__main__":
    main()