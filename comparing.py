import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import math
from maze import MazeGenerator
from LookingGood import AStarDWAAgent
from dynamicwindow2D import dwa_control, Config as DWAConfig, RobotType, motion
from customastar import CustomAStarAgent

def run_astar_dwa(start, goal, static_obstacles, config):
    agent = AStarDWAAgent(static_obstacles, config.resolution, config.robot_radius, config.max_speed)
    path = agent.plan_path(start[0], start[1], goal[0], goal[1])
    if path:
        start_time = time.time()
        start_state = list(start) + [math.pi / 8.0, 0.0, 0.0]
        goal_list = list(goal)
        trajectory = agent.move_to_goal(start_state, goal_list, path)
        end_time = time.time()
        return len(trajectory), end_time - start_time
    return None, None

def run_dwa(start, goal, obstacles, config):
    x = np.array(start + [np.pi/8, 0.0, 0.0])
    trajectory = [x[:2]]
    goal = np.array(goal)
    ob = np.array([[obs[0], obs[1]] for obs in obstacles])
    
    start_time = time.time()
    while True:
        u, _ = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)
        trajectory.append(x[:2])
        
        dist_to_goal = np.linalg.norm(x[:2] - goal)
        if dist_to_goal <= config.robot_radius:
            break
    end_time = time.time()
    
    return len(trajectory), end_time - start_time

def run_custom_astar(start, goal, obstacles, config):
    agent = CustomAStarAgent(config.resolution, config.robot_radius)
    start_time = time.time()
    path = agent.plan_path(start, goal, obstacles)
    if path:
        trajectory = agent.move_to_goal(list(start), list(goal), path, obstacles)
        end_time = time.time()
        return len(trajectory), end_time - start_time
    return None, None

def run_tests():
    maze_gen = MazeGenerator(50, 50)
    results = []
    
    for difficulty in range(1, 11):
        maze = maze_gen.get_maze(difficulty)
        obstacles = maze_gen.maze_to_obstacles(maze)
        obstacle_count = len(obstacles)
        start = maze_gen.start
        goal = maze_gen.goal
        
        config = DWAConfig()
        config.robot_type = RobotType.circle
        config.resolution = 0.5
        config.robot_radius = 0.5
        config.max_speed = 1.0
        
        for _ in range(10):  # 10 trials per maze per algorithm
            # A* DWA
            path_length, time_taken = run_astar_dwa(start, goal, obstacles, config)
            if path_length:
                results.append(["A* DWA", difficulty, obstacle_count, path_length, time_taken])
            
            # DWA
            path_length, time_taken = run_dwa(start, goal, obstacles, config)
            results.append(["DWA", difficulty, obstacle_count, path_length, time_taken])
            
            # Custom A*
            path_length, time_taken = run_custom_astar(start, goal, obstacles, config)
            if path_length:
                results.append(["Custom A*", difficulty, obstacle_count, path_length, time_taken])
    
    return results

def save_results_to_csv(results, filename="algorithm_comparison_results.csv"):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Maze Difficulty", "Obstacle Count", "Path Length", "Time to Goal"])
        writer.writerows(results)

if __name__ == "__main__":
    results = run_tests()
    save_results_to_csv(results)
    print(f"Results saved to algorithm_comparison_results.csv")