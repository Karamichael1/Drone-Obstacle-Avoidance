import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from heapq import heappush, heappop
import time

class Node:
    def __init__(self, pos, g_cost, h_cost, parent):
        self.pos = pos
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost or (self.f_cost == other.f_cost and self.h_cost < other.h_cost)

class Config:
    def __init__(self):
        self.max_speed = 1.0
        self.min_speed = -0.5
        self.max_yawrate = 40.0 * math.pi / 180.0
        self.max_accel = 0.2
        self.max_dyawrate = 40.0 * math.pi / 180.0
        self.dt = 0.1
        self.robot_radius = 1.5

def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

class CustomAStar:
    def __init__(self, grid_size, robot_radius, map_width, map_height):
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        self.map_width = map_width
        self.map_height = map_height
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]
        self.obstacle_influence_distance = 2 * robot_radius
        self.config = Config()

    def plan(self, start, goal, obstacles, visualize=False):
        self.obstacle_set = set((round(obs[0]/self.grid_size), round(obs[1]/self.grid_size)) 
                                for obs in obstacles)
        start = (round(start[0] / self.grid_size), round(start[1] / self.grid_size))
        goal = (round(goal[0] / self.grid_size), round(goal[1] / self.grid_size))

        obstacle_set = set()
        for obs in obstacles:
            x, y, radius = obs
            grid_radius = math.ceil((radius + self.robot_radius) / self.grid_size)
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    if dx*dx + dy*dy <= grid_radius*grid_radius:
                        grid_x = round(x / self.grid_size) + dx
                        grid_y = round(y / self.grid_size) + dy
                        obstacle_set.add((grid_x, grid_y))

        open_list = []
        closed_set = set()
        node_dict = {}

        start_node = Node(start, 0, self.heuristic(start, goal, obstacles), None)
        heappush(open_list, start_node)
        node_dict[start] = start_node

        if visualize:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(0, max(start[0], goal[0]) + 10)
            ax.set_ylim(0, max(start[1], goal[1]) + 10)
            ax.set_aspect('equal')
            
            # Plot obstacles
            for obs in obstacles:
                circle = plt.Circle((obs[0]/self.grid_size, obs[1]/self.grid_size), obs[2]/self.grid_size, fill=True, color='red', alpha=0.5)
                ax.add_artist(circle)
            
            # Plot start and goal
            ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
            ax.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')
            ax.legend()

        while open_list:
            current = heappop(open_list)
            if current.pos == goal:
                return self.reconstruct_path(current)

            for neighbor in self.get_neighbors(current.pos):
                if not self.is_valid(neighbor):
                    continue

            if current.pos == goal:
                path = self.reconstruct_path(current)
                if visualize:
                    path_x, path_y = zip(*path)
                    ax.plot(path_x, path_y, 'g-', linewidth=2)
                    plt.draw()
                    plt.pause(0.001)
                return path

            closed_set.add(current.pos)

            if visualize:
                ax.plot(current.pos[0], current.pos[1], 'yo', markersize=5)
                plt.draw()
                plt.pause(0.001)

            for direction in self.directions:
                neighbor_pos = (current.pos[0] + direction[0], current.pos[1] + direction[1])

                if (neighbor_pos in obstacle_set or neighbor_pos in closed_set):
                    continue

                g_cost = current.g_cost + (1.4 if abs(direction[0] + direction[1]) == 2 else 1)
                h_cost = self.heuristic(neighbor_pos, goal, obstacles)
                neighbor = Node(neighbor_pos, g_cost, h_cost, current)

                if neighbor_pos not in node_dict or g_cost < node_dict[neighbor_pos].g_cost:
                    node_dict[neighbor_pos] = neighbor
                    heappush(open_list, neighbor)

                    if visualize:
                        ax.plot(neighbor_pos[0], neighbor_pos[1], 'co', markersize=3)

        if visualize:
            plt.close(fig)
        return None  # No path found

    
    def heuristic(self, node, goal, obstacles):
        base_cost = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        
        penalty = 0
        node_pos = np.array(node)
        for obs in obstacles:
            obs_pos = np.array(obs[:2])  # Extract only x and y coordinates
            obs_radius = obs[2]
            distance = np.linalg.norm(node_pos - obs_pos) - obs_radius - self.robot_radius
            if distance < self.obstacle_influence_distance:
                penalty += (self.obstacle_influence_distance - distance) / self.obstacle_influence_distance
        
        penalty_weight = 10.0
        return base_cost + penalty_weight * penalty
    def get_neighbors(self, pos):
        x, y = pos
        return [(x+dx, y+dy) for dx, dy in self.directions 
                if self.is_valid((x+dx, y+dy))]
    def is_valid(self, pos):
        x, y = pos
        return (0 <= x < self.map_width and 
                0 <= y < self.map_height and 
                pos not in self.obstacle_set)
        
    def reconstruct_path(self, node):
        path = []
        while node:
            path.append((node.pos[0] * self.grid_size, node.pos[1] * self.grid_size))
            node = node.parent
        return list(reversed(path))
    def simulate_path_traversal(self, start, goal, path, obstacles, visualize=False):
        if not path:
            return np.array([start])

        # Initialize robot state [x, y, yaw, velocity, angular_velocity]
        x = np.array([start[0], start[1], math.pi / 8.0, 0.0, 0.0])
        trajectory = np.array([x])
        
        target_ind = 0
        
        if visualize:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(0, self.map_width)
            ax.set_ylim(0, self.map_height)
            
            # Plot obstacles
            for obs in obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2], fill=True, color='red', alpha=0.5)
                ax.add_artist(circle)
            
            # Plot path
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'b--', alpha=0.7)
            
            # Plot start and goal
            ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
            ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
            ax.legend()

        while target_ind < len(path):
            tx, ty = path[target_ind]
            dist_to_target = math.hypot(x[0] - tx, x[1] - ty)

            if dist_to_target <= self.config.robot_radius:
                target_ind += 1
                if target_ind >= len(path):
                    break

            # Calculate steering angle
            target_angle = math.atan2(ty - x[1], tx - x[0])
            steering = self.proportional_control(target_angle, x[2])

            # Apply control
            u = np.array([self.config.max_speed, steering])
            x = motion(x, u, self.config.dt)
            trajectory = np.vstack((trajectory, x))

            if visualize:
                plt.cla()
                # Redraw obstacles
                for obs in obstacles:
                    circle = plt.Circle((obs[0], obs[1]), obs[2], fill=True, color='red', alpha=0.5)
                    ax.add_artist(circle)
                
                # Plot path and current position
                ax.plot(path_x, path_y, 'b--', alpha=0.7)
                ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-')
                ax.plot(x[0], x[1], 'go', markersize=10)
                ax.plot(goal[0], goal[1], 'ro', markersize=10)
                
                ax.set_xlim(0, self.map_width)
                ax.set_ylim(0, self.map_height)
                plt.pause(0.001)

        if visualize:
            plt.close()

        return trajectory

    def proportional_control(self, target_angle, current_angle):
        return self.config.max_yawrate * math.atan2(math.sin(target_angle - current_angle),
                                                   math.cos(target_angle - current_angle))

class CustomAStarAgent:
    def __init__(self, grid_size, robot_radius, map_width, map_height):
        self.astar = CustomAStar(grid_size, robot_radius, map_width, map_height)

    def plan_path(self, start, goal, obstacles, visualize=False):
        #print(f"Planning path - Start: {start}, Goal: {goal}, Number of obstacles: {len(obstacles)}")
        return self.astar.plan(start, goal, obstacles, visualize)

    def move_to_goal(self, start, goal, path, obstacles, visualize=False):
        return self.astar.simulate_path_traversal(start, goal, path, obstacles, visualize)