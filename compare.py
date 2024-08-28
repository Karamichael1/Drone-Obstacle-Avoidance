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
    fig, ax = plt.subplots(figsize=(15, 10))
    
    speeds = list(results.keys())
    algorithms = ['Custom A*', 'DWA', 'A* + DWA']
    colors = ['r', 'g', 'b']
    
    for i, algorithm in enumerate(algorithms):
        execution_times = []
        for speed in speeds:
            metrics = results[speed][i]
            if metrics and metrics.success:
                execution_times.append(metrics.execution_times[-1])
            else:
                execution_times.append(float('nan'))
        
        ax.plot(speeds, execution_times, f'-{colors[i]}o', label=algorithm)
        
        for j, time in enumerate(execution_times):
            if math.isnan(time):
                ax.text(speeds[j], ax.get_ylim()[1], 'Failed', 
                         ha='center', va='bottom', color=colors[i], rotation=90)
    
    ax.set_xlabel('Speed')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Algorithm Performance Comparison')
    ax.legend()
    ax.grid(True)
    
    fig.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    plt.xlabel('Speed')
    plt.ylabel('Execution Time (s)')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
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
    
    for speed in np.arange(0.5, 1, 0.5):
        print(f"Running simulation with speed: {speed}")
        
        custom_astar_metrics = run_with_timeout(run_custom_astar, (start, goal, obstacles, speed), 60)
        dwa_metrics = run_with_timeout(run_dwa, (start, goal, np.array(obstacles)[:, :2], speed), 60)
        astar_dwa_metrics = run_with_timeout(run_astar_dwa, (start, goal, obstacles, speed), 60)
        
        results[speed] = [custom_astar_metrics, dwa_metrics, astar_dwa_metrics]
        
        for name, metrics in zip(['Custom A*', 'DWA', 'A* + DWA'], results[speed]):
            if metrics and metrics.success:
                print(f"{name} - Time: {metrics.execution_times[-1]:.2f}s, Path Length: {metrics.path_length:.2f}")
            else:
                print(f"{name} - Failed to complete within 60 seconds or encountered an error")
    
    plot_results(results)

if __name__ == "__main__":
    main()