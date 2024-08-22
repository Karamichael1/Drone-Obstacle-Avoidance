import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Import A* related classes and functions
from AStar import AStarPlanner

# Import DWA related classes and functions
from dynamicwindow2D import Config, RobotType, dwa_control, motion, plot_robot, plot_arrow

show_animation = True
import numpy as np
from scipy.spatial import distance

class AStarDWAAgent:
    def __init__(self, static_obstacles, resolution, robot_radius):
        self.static_obstacles = static_obstacles
        ox = [obs[0] for obs in static_obstacles]
        oy = [obs[1] for obs in static_obstacles]
        self.a_star = AStarPlanner(ox, oy, resolution, robot_radius)
        self.dwa_config = Config()
        self.dwa_config.robot_type = RobotType.circle
        self.dwa_config.robot_radius = robot_radius
        self.dwa_config.max_speed = 1.0  # Adjust this value to control the agent's speed

    def plan_path(self, sx, sy, gx, gy):
        rx, ry = self.a_star.planning(sx, sy, gx, gy)
        return list(reversed(rx)), list(reversed(ry))
    
    def move_to_goal(self, start, goal, path, dynamic_obstacles):
        x = np.array(start + [math.pi / 8.0, 0.0, 0.0])  # initial state [x, y, yaw, v, omega]
        trajectory = np.array(x)
        rx, ry = path
        
        target_ind = 0
        time = 0
        while True:
            if target_ind < len(rx) - 1:
                local_goal = [rx[target_ind], ry[target_ind]]
            else:
                local_goal = goal
            
            # Update dynamic obstacles
            current_obstacles = self.update_obstacles(time, dynamic_obstacles)
            
            # Convert obstacles to the format expected by dwa_control
            dwa_obstacles = np.array([[obs[0], obs[1]] for obs in current_obstacles])
            
            # Check for nearby dynamic obstacles
            nearby_obstacle = self.check_nearby_obstacles(x, current_obstacles)
            
            if nearby_obstacle:
                # Use DWA for local avoidance
                u, predicted_trajectory = dwa_control(x, self.dwa_config, local_goal, dwa_obstacles)
            else:
                # Move towards A* path
                direction = math.atan2(local_goal[1] - x[1], local_goal[0] - x[0])
                u = [self.dwa_config.max_speed, 0.0]  # [speed, yaw_rate]
            
            x = motion(x, u, self.dwa_config.dt)
            trajectory = np.vstack((trajectory, x))
            
            # Update target index
            dist_to_target = math.hypot(x[0] - rx[target_ind], x[1] - ry[target_ind])
            if dist_to_target <= self.dwa_config.robot_radius:
                target_ind += 1
            
            if show_animation:
                plt.cla()
                plt.plot(rx, ry, "-r", linewidth=2)
                if nearby_obstacle:
                    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
                plt.plot(x[0], x[1], "xr")
                plt.plot(goal[0], goal[1], "xb")
                self.plot_obstacles(current_obstacles)
                plot_robot(x[0], x[1], x[2], self.dwa_config)
                plot_arrow(x[0], x[1], x[2])
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.0001)
            
            # Check if goal is reached
            dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
            if dist_to_goal <= self.dwa_config.robot_radius:
                print("Goal!!")
                break
            
            time += self.dwa_config.dt
        
        return trajectory

    def update_obstacles(self, time, dynamic_obstacles):
        current_obstacles = self.static_obstacles.copy()
        for obs in dynamic_obstacles:
            x = obs['x'] + obs['vx'] * time
            y = obs['y'] + obs['vy'] * time
            current_obstacles.append((x, y, obs['radius']))
        return current_obstacles

    def check_nearby_obstacles(self, x, obstacles):
        robot_position = x[:2]
        for obs in obstacles:
            if distance.euclidean(robot_position, obs[:2]) < self.dwa_config.robot_radius + obs[2] + 2.0:
                return True
        return False

    def plot_obstacles(self, obstacles):
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], fill=False)
            plt.gca().add_artist(circle)

def main():
    print("AStarDWAAgent simulation start")
    
    # Define map size
    map_size = 50.0
    
    # Start and goal position
    sx, sy = 5.0, 5.0
    gx, gy = 45.0, 45.0
    
    grid_size = 2.0
    robot_radius = 1.0

    # Set fixed static obstacles
    static_obstacles = [
        (10.0, 10.0, 2.0),
        (20.0, 20.0, 2.0),
        (30.0, 30.0, 2.0),
        (40.0, 40.0, 2.0),
        (15.0, 35.0, 2.0),
        (35.0, 15.0, 2.0)
    ]

    # Set fixed dynamic obstacles
    dynamic_obstacles = [
        {'x': 25.0, 'y': 25.0, 'vx': 0.5, 'vy': 0.5, 'radius': 1.5},
        {'x': 35.0, 'y': 20.0, 'vx': -0.3, 'vy': 0.7, 'radius': 2.0},
        {'x': 15.0, 'y': 30.0, 'vx': 0.6, 'vy': -0.2, 'radius': 1.8},
    ]
    
    print(f"Start position: ({sx}, {sy})")
    print(f"Goal position: ({gx}, {gy})")
    print(f"Static obstacles: {static_obstacles}")
    print(f"Dynamic obstacles: {dynamic_obstacles}")

    agent = AStarDWAAgent(static_obstacles, grid_size, robot_radius)
    
    # Calculate the A* path
    path = agent.plan_path(sx, sy, gx, gy)
    
    if not path[0]:
        print("No path found. Exiting.")
        return

    # Start moving along the path
    trajectory = agent.move_to_goal([sx, sy], [gx, gy], path, dynamic_obstacles)
    
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()

if __name__ == "__main__":
    main()