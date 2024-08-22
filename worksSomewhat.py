import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import time
import random

# Import necessary classes and functions from DStarLite
from DStarLite import DStarLite, Node, compare_coordinates

# Import necessary classes and functions from dynamicwindow2D
from dynamicwindow2D import Config, RobotType, dwa_control, plot_robot, plot_arrow

show_animation = True

class DynamicObstacle:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = random.uniform(0, 2 * math.pi)

    def move(self):
        self.x += self.speed * math.cos(self.direction)
        self.y += self.speed * math.sin(self.direction)
        # Change direction occasionally
        if random.random() < 0.1:
            self.direction = random.uniform(0, 2 * math.pi)

class OptimizedIntegratedPlanner:
    def __init__(self, start: Node, goal: Node, ox: List[float], oy: List[float], map_size: int):
        self.start = start
        self.goal = goal
        self.ox = ox
        self.oy = oy
        self.map_size = map_size
        
        # Initialize D* Lite
        self.dstar = DStarLite(ox, oy)
        self.dstar.initialize(start, goal)
        
        # Initialize DWA config
        self.config = Config()
        self.config.robot_type = RobotType.circle
        self.config.ob = np.column_stack((ox, oy))
        
        # Set up the initial state for DWA
        self.x = np.array([start.x, start.y, math.atan2(goal.y - start.y, goal.x - start.x), 0.0, 0.0])
        
        # Compute the global path once
        self.dstar.compute_shortest_path()
        self.global_path = self.dstar.compute_current_path()
        
        # Flag to indicate if we're using DWA
        self.using_dwa = False
        
        # Threshold distance to consider a dynamic obstacle as potentially colliding
        self.collision_threshold = 3.0  # Increased for earlier detection

        # Dynamic obstacles
        self.dynamic_obstacles = self.generate_dynamic_obstacles(5)  # Generate 5 dynamic obstacles

        # Initialize the plot
        if show_animation:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-5, map_size + 5)
            self.ax.set_ylim(-5, map_size + 5)

    def generate_dynamic_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                x = random.uniform(0, self.map_size)
                y = random.uniform(0, self.map_size)
                if not self.is_position_obstacle(x, y):
                    speed = random.uniform(0.05, 0.1)  # Slower speed
                    obstacles.append(DynamicObstacle(x, y, speed))
                    break
        return obstacles

    def is_position_obstacle(self, x, y):
        for ox, oy in zip(self.ox, self.oy):
            if math.hypot(ox - x, oy - y) < self.config.robot_radius * 2:
                return True
        return False

    def potential_collision_with_dynamic_obstacle(self, current_pos: np.ndarray, next_pos: np.ndarray) -> bool:
        """Check if there's a potential collision with a dynamic obstacle."""
        for ob in self.dynamic_obstacles:
            d_current = math.hypot(current_pos[0] - ob.x, current_pos[1] - ob.y)
            d_next = math.hypot(next_pos[0] - ob.x, next_pos[1] - ob.y)
            if d_current < self.collision_threshold or d_next < self.collision_threshold:
                return True
        return False
    
    def run(self):
        trajectory = np.array(self.x)
        
        while True:
            # Move dynamic obstacles
            for ob in self.dynamic_obstacles:
                ob.move()
                # Ensure dynamic obstacles don't go out of bounds
                ob.x = max(0, min(ob.x, self.map_size))
                ob.y = max(0, min(ob.y, self.map_size))

            # Get the next waypoint from the global path
            if len(self.global_path) > 1:
                next_waypoint = self.global_path[1]
                local_goal = np.array([next_waypoint.x, next_waypoint.y])
            else:
                local_goal = np.array([self.goal.x, self.goal.y])
            
            # Calculate the next position if we move towards the next waypoint
            direction = math.atan2(next_waypoint.y - self.x[1], next_waypoint.x - self.x[0])
            speed = min(self.config.max_speed, self.distance_to_point(self.x, next_waypoint))
            next_x = self.update_state(self.x.copy(), [speed, 0.0], self.config.dt)
            
            # Check for potential collision with dynamic obstacles
            if self.potential_collision_with_dynamic_obstacle(self.x[:2], next_x[:2]):
                # Use DWA for local planning
                dwa_ob = np.vstack((self.config.ob, np.array([[ob.x, ob.y] for ob in self.dynamic_obstacles])))
                u, predicted_trajectory = dwa_control(self.x, self.config, local_goal, dwa_ob)
                self.x = self.update_state(self.x, u, self.config.dt)
                self.using_dwa = True
            else:
                # Move directly towards the next waypoint
                self.x = next_x
                predicted_trajectory = np.array([self.x])
                self.using_dwa = False
            
            trajectory = np.vstack((trajectory, self.x))
            
            # Check if we've reached the current waypoint
            if len(self.global_path) > 1 and self.distance_to_point(self.x, next_waypoint) < self.config.robot_radius:
                self.global_path.pop(0)
            
            # Visualization
            if show_animation:
                self.plot(predicted_trajectory, trajectory)
            
            # Check if goal is reached
            dist_to_goal = self.distance_to_point(self.x, self.goal)
            if dist_to_goal <= self.config.robot_radius:
                print("Goal!!")
                break
        
        print("Done")
        if show_animation:
            plt.show()

    @staticmethod
    def update_state(x, u, dt):
        """Update the robot state."""
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    @staticmethod
    def distance_to_point(x, point):
        """Calculate the distance between the robot and a point."""
        return math.hypot(x[0] - point.x, x[1] - point.y)

    def plot(self, predicted_trajectory, trajectory):
        self.ax.clear()
        # Plot static obstacles
        self.ax.plot(self.ox, self.oy, ".k")
        # Plot dynamic obstacles
        for ob in self.dynamic_obstacles:
            self.ax.plot(ob.x, ob.y, "or")
        # Plot start and goal
        self.ax.plot(self.start.x, self.start.y, "og")
        self.ax.plot(self.goal.x, self.goal.y, "xb")
        # Plot global path
        path_x = [node.x for node in self.global_path]
        path_y = [node.y for node in self.global_path]
        self.ax.plot(path_x, path_y, "-c")
        # Plot robot
        plot_robot(self.x[0], self.x[1], self.x[2], self.config)
        # Plot arrow if using DWA
        if self.using_dwa:
            plot_arrow(self.x[0], self.x[1], self.x[2])
        # Plot trajectory
        self.ax.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        
        self.ax.set_xlim(-5, self.map_size + 5)
        self.ax.set_ylim(-5, self.map_size + 5)
        self.ax.grid(True)
        plt.pause(0.001)

def main():
    map_size = 60
    # Set start and goal positions
    sx, sy = 5, 5
    gx, gy = 55, 55

    # Set obstacle positions
    ox, oy = [], []
    for i in range(-5, map_size + 5):
        ox.append(i)
        oy.append(-5)
        ox.append(i)
        oy.append(map_size + 5)
    for i in range(-5, map_size + 5):
        ox.append(-5)
        oy.append(i)
        ox.append(map_size + 5)
        oy.append(i)
    
    # Add some obstacles in the middle
    for i in range(20, 40):
        ox.append(i)
        oy.append(30)
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)

    # Initialize and run the optimized integrated planner
    planner = OptimizedIntegratedPlanner(Node(sx, sy), Node(gx, gy), ox, oy, map_size)
    planner.run()

if __name__ == "__main__":
    main()