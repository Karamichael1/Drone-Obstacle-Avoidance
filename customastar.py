import math
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

class Node:
    def __init__(self, pos, g_cost, h_cost, parent):
        self.pos = pos
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class CustomAStar:
    def __init__(self, grid_size, robot_radius):
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def plan(self, start, goal, obstacles):
        start = (round(start[0] / self.grid_size), round(start[1] / self.grid_size))
        goal = (round(goal[0] / self.grid_size), round(goal[1] / self.grid_size))

        obstacle_set = set()
        for obs in obstacles:
            x, y, radius = obs
            for dx in range(-int(radius/self.grid_size)-1, int(radius/self.grid_size)+2):
                for dy in range(-int(radius/self.grid_size)-1, int(radius/self.grid_size)+2):
                    if dx*dx + dy*dy <= (radius/self.grid_size)**2:
                        obstacle_set.add((round(x/self.grid_size)+dx, round(y/self.grid_size)+dy))

        open_list = []
        closed_set = set()

        start_node = Node(start, 0, self.heuristic(start, goal), None)
        heappush(open_list, start_node)

        while open_list:
            current = heappop(open_list)

            if current.pos == goal:
                return self.reconstruct_path(current)

            closed_set.add(current.pos)

            for direction in self.directions:
                neighbor_pos = (current.pos[0] + direction[0], current.pos[1] + direction[1])

                if (neighbor_pos in obstacle_set or
                    neighbor_pos in closed_set or
                    not self.is_valid(neighbor_pos)):
                    continue

                g_cost = current.g_cost + (1.4 if abs(direction[0] + direction[1]) == 2 else 1)
                h_cost = self.heuristic(neighbor_pos, goal)
                neighbor = Node(neighbor_pos, g_cost, h_cost, current)

                if neighbor not in open_list:
                    heappush(open_list, neighbor)
                else:
                    # Update the neighbor if this path is better
                    idx = open_list.index(neighbor)
                    if open_list[idx].g_cost > g_cost:
                        open_list[idx].g_cost = g_cost
                        open_list[idx].f_cost = g_cost + h_cost
                        open_list[idx].parent = current

        return None # No path found

    def heuristic(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def is_valid(self, pos):
        return pos[0] >= 0 and pos[1] >= 0  # Assuming non-negative grid coordinates

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append((node.pos[0] * self.grid_size, node.pos[1] * self.grid_size))
            node = node.parent
        return list(reversed(path))

class Config:
    def __init__(self):
        self.max_speed = 1.0
        self.min_speed = -0.5
        self.max_yawrate = 40.0 * math.pi / 180.0
        self.max_accel = 0.2
        self.max_dyawrate = 40.0 * math.pi / 180.0
        self.dt = 0.1
        self.robot_radius = 1.0

def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def plot_arrow(x, y, yaw, length=0.5, width=0.1):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_width=width, head_length=width)

class CustomAStarAgent:
    def __init__(self, grid_size, robot_radius):
        self.astar = CustomAStar(grid_size, robot_radius)  # Fixed: Use CustomAStar instead of CustomAStarAgent
        self.config = Config()

    def plan_path(self, start, goal, obstacles):
        return self.astar.plan(start, goal, obstacles)

    def move_to_goal(self, start, goal, path, obstacles):
        x = np.array(start + [math.pi / 8.0, 0.0, 0.0])
        trajectory = np.array(x)

        target_ind = 0
        while True:
            if target_ind < len(path) - 1:
                tx, ty = path[target_ind]
            else:
                tx, ty = goal

            dist_to_target = math.hypot(x[0] - tx, x[1] - ty)

            if dist_to_target <= self.config.robot_radius:
                target_ind += 1

            if target_ind >= len(path):
                break

            # Simple proportional control for steering
            target_angle = math.atan2(ty - x[1], tx - x[0])
            steering = self.proportional_control(target_angle, x[2])

            # Move towards the target at constant speed
            u = np.array([self.config.max_speed, steering])
            x = motion(x, u, self.config.dt)
            trajectory = np.vstack((trajectory, x))

            if self.show_animation(x, goal, path, obstacles, trajectory):
                plt.pause(0.001)

        print("Goal reached!")
        return trajectory

    def proportional_control(self, target_angle, current_angle):
        return self.config.max_yawrate * math.atan2(math.sin(target_angle - current_angle),
                                                    math.cos(target_angle - current_angle))

    def show_animation(self, x, goal, path, obstacles, trajectory):
        plt.cla()
        # Plot path
        plt.plot([p[0] for p in path], [p[1] for p in path], "-r")
        # Plot trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-b")
        # Plot robot
        plot_arrow(x[0], x[1], x[2])
        # Plot goal
        plt.plot(goal[0], goal[1], "xg")
        # Plot obstacles
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], fill=True)
            plt.gca().add_artist(circle)
        plt.axis("equal")
        plt.grid(True)
        return True

def main():
    print("CustomAStarAgent simulation start")

    # Define simulation parameters
    start = (5.0, 5.0)
    goal = (45.0, 45.0)
    grid_size = 2.0
    robot_radius = 1.0

    obstacles = [
        (10.0, 10.0, 1.0),
        (20.0, 20.0, 1.0),
        (30.0, 26.0, 1.0),
        (40.0, 42.0, 1.0),
        (15.0, 35.0, 1.0),
        (35.0, 15.0, 1.0)
    ]

    agent = CustomAStarAgent(grid_size, robot_radius)

    path = agent.plan_path(start, goal, obstacles)
    
    if not path:
        print("No path found. Exiting.")
        return

    trajectory = agent.move_to_goal(list(start), list(goal), path, obstacles)

    plt.show()

if __name__ == "__main__":
    main()