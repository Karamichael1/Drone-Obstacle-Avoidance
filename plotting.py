import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance
import time

from dynamicwindow2D import Config, RobotType, dwa_control, motion, plot_arrow
from worksSomewhat2 import AStarDWAAgent
from customastar import CustomAStar

show_animation = False  # Set to True if you want to see the animations in real-time

def run_astar(start, goal, obstacles, resolution, robot_radius, speed, dt):
    astar = CustomAStar(resolution, robot_radius)
    path = astar.plan(start, goal, obstacles)
    if path is None or len(path) < 2:
        print("A* couldn't find a path.")
        return None
    
    print(f"A* found a path with {len(path)} points.")
    
    x = np.array([start[0], start[1], math.atan2(path[1][1]-start[1], path[1][0]-start[0]), 0.0, 0.0])
    trajectory = [x]
    
    target_idx = 1
    sim_time = 0
    max_iterations = 1000  # Set a maximum number of iterations
    
    for iteration in range(max_iterations):
        if target_idx >= len(path):
            print(f"A* reached the end of the path after {iteration} iterations.")
            break
        
        dx = path[target_idx][0] - x[0]
        dy = path[target_idx][1] - x[1]
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        if distance_to_target < 0.1:  # If we're very close to the current target point
            target_idx += 1
            print(f"Moving to next target point. {len(path) - target_idx} points remaining.")
            continue
        
        target_angle = math.atan2(dy, dx)
        
        # Simple proportional control for angular velocity
        angle_diff = target_angle - x[2]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]
        omega = 5.0 * angle_diff  # You can adjust this coefficient
        
        # Move towards the target point
        if distance_to_target < speed * dt:
            v = distance_to_target / dt
        else:
            v = speed
        
        # Update state
        x = motion(x, [v, omega], dt)
        trajectory.append(x)
        
        # Check if goal is reached
        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= robot_radius:
            print(f"A* reached the goal after {iteration} iterations.")
            break
        
        sim_time += dt
        time.sleep(dt)  # This will make the simulation run in real-time
    
    else:
        print(f"A* reached maximum iterations ({max_iterations}) without reaching the goal.")
    
    return np.array(trajectory)

def run_dwa(start, goal, obstacles, config, speed):
    config.max_speed = speed
    x = np.array([start[0], start[1], math.pi / 8.0, 0.0, 0.0])
    trajectory = np.array([x])
    ob = np.array([[obs[0], obs[1]] for obs in obstacles])
    
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)
        trajectory = np.vstack((trajectory, x))
        
        if math.hypot(x[0] - goal[0], x[1] - goal[1]) <= config.robot_radius:
            break
    
    return trajectory

def run_astar_dwa(start, goal, obstacles, resolution, robot_radius, speed):
    agent = AStarDWAAgent(obstacles, resolution, robot_radius)
    agent.dwa_config.max_speed = speed
    path = agent.plan_path(start[0], start[1], goal[0], goal[1])
    if not path[0]:
        print("A*-DWA couldn't find a path.")
        return None
    
    # Simulate movement without animation
    x = np.array([start[0], start[1], math.pi / 8.0, 0.0, 0.0])
    trajectory = [x]
    rx, ry = path
    target_ind = 0
    
    while True:
        if target_ind < len(rx) - 1:
            local_goal = [rx[target_ind], ry[target_ind]]
        else:
            local_goal = goal
        
        dwa_obstacles = np.array([[obs[0], obs[1]] for obs in obstacles])
        u, _ = dwa_control(x, agent.dwa_config, local_goal, dwa_obstacles)
        x = motion(x, u, agent.dwa_config.dt)
        trajectory.append(x)
        
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= agent.dwa_config.robot_radius:
            break
        
        if target_ind < len(rx) - 1:
            dist_to_target = math.hypot(x[0] - rx[target_ind], x[1] - ry[target_ind])
            if dist_to_target <= agent.dwa_config.robot_radius:
                target_ind += 1

    return np.array(trajectory)

def plot_results(results, obstacles, start, goal, execution_times, filename):
    plt.figure(figsize=(12, 8))
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], fill=True, color='gray')
        plt.gca().add_artist(circle)

    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    colors = ['b', 'g', 'r']
    for (name, trajectory), color in zip(results.items(), colors):
        if trajectory is not None:
            plt.plot(trajectory[:, 0], trajectory[:, 1], color, label=f"{name} ({execution_times[name]:.2f}s)")

    plt.legend()
    plt.title("Algorithm Comparison")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
 
    text = "\n".join([f"{name}: {time:.2f}s" for name, time in execution_times.items()])
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, verticalalignment='top', fontsize=9)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    

def main():
    print("Algorithm comparison simulation start")
    
    map_size = 50.0
    start = (5.0, 5.0)
    goal = (45.0, 45.0)
    grid_size = 2.0
    robot_radius = 1.0
    speed = 1.0  # Set the speed for all algorithms
    dt = 0.1  # Time step

    obstacles = [
        (10.0, 10.0, 1.0),
        (20.0, 20.0, 1.0),
        (30.0, 26.0, 1.0),
        (40.0, 42.0, 1.0),
        (15.0, 35.0, 1.0),
        (35.0, 15.0, 1.0)
    ]

    print(f"Start position: {start}")
    print(f"Goal position: {goal}")
    print(f"Obstacles: {obstacles}")
    print(f"Speed: {speed}")

    dwa_config = Config()
    dwa_config.robot_type = RobotType.circle
    dwa_config.robot_radius = robot_radius
    dwa_config.dt = dt

    algorithms = {
        "A*": lambda: run_astar(start, goal, obstacles, grid_size, robot_radius, speed, dt),
        "DWA": lambda: run_dwa(start, goal, obstacles, dwa_config, speed),
        "A*-DWA": lambda: run_astar_dwa(start, goal, obstacles, grid_size, robot_radius, speed)
    }

    results = {}
    for name, algorithm in algorithms.items():
        print(f"Running {name}...")
        start_time = time.time()
        trajectory = algorithm()
        end_time = time.time()
        results[name] = trajectory
        if trajectory is None:
            print(f"{name} failed to find a path.")
        else:
            print(f"{name} completed in {end_time - start_time:.2f} seconds")

    # Plot results and create animation
    plot_results(results, obstacles, start, goal, "algorithm_comparison.png")
   

    # Save numerical results
    with open("algorithm_results.txt", "w") as f:
        for name, trajectory in results.items():
            if trajectory is not None:
                path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
                execution_time = end_time - start_time
                f.write(f"{name}:\n")
                f.write(f"  Path length: {path_length:.2f}\n")
                f.write(f"  Number of steps: {len(trajectory)}\n")
                f.write(f"  Execution time: {execution_time:.2f} seconds\n\n")
            else:
                f.write(f"{name}: Failed to find a path\n\n")

    print("Results saved to 'algorithm_results.txt'")
    print("Plots and animations saved to 'algorithm_comparison.png' and 'algorithm_comparison.mp4'")

if __name__ == "__main__":
    main()