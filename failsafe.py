import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from AStar import AStarPlanner

from dynamicwindow2D import Config, RobotType, dwa_control, motion, plot_arrow

show_animation = True
import numpy as np
from scipy.spatial import distance

class AStarDWAAgent:
    def __init__(self, static_obstacles, resolution, robot_radius):
        self.static_obstacles = [(obs[0], obs[1], obs[2] + 0.5) for obs in static_obstacles]  # Add 0.5 unit padding
        ox = [obs[0] for obs in self.static_obstacles]
        oy = [obs[1] for obs in self.static_obstacles]
        self.a_star = AStarPlanner(ox, oy, resolution, robot_radius)
        self.dwa_config = Config()
        self.dwa_config.robot_type = RobotType.circle
        self.dwa_config.robot_radius = robot_radius
        self.dwa_config.max_speed = 1.0 

    def plan_path(self, sx, sy, gx, gy):
        try:
            rx, ry = self.a_star.planning(sx, sy, gx, gy)
            if rx and ry:
                return list(reversed(rx)), list(reversed(ry))
            else:
                print("A* failed to find a path. Switching to pure DWA.")
                return None
        except Exception as e:
            print(f"Error in A* planning: {str(e)}. Switching to pure DWA.")
            return None

    def move_to_goal(self, start, goal, path, dynamic_obstacles):
        x = np.array(start + [math.pi / 8.0, 0.0, 0.0])
        trajectory = np.array(x)
        
        if path:
            rx, ry = path
        else:
            rx, ry = [], []  # Empty path for pure DWA
        
        target_ind = 0
        time = 0
        while True:
            if path and target_ind < len(rx) - 1:
                local_goal = [rx[target_ind], ry[target_ind]]
            else:
                local_goal = goal
            
            current_obstacles = self.update_obstacles(time, dynamic_obstacles)
            dwa_obstacles = np.array([[obs[0], obs[1]] for obs in current_obstacles])
            
            u, predicted_trajectory = dwa_control(x, self.dwa_config, local_goal, dwa_obstacles)
            
            x = motion(x, u, self.dwa_config.dt)
            trajectory = np.vstack((trajectory, x))
            
            if path:
                dist_to_target = math.hypot(x[0] - rx[target_ind], x[1] - ry[target_ind])
                if dist_to_target <= self.dwa_config.robot_radius:
                    target_ind += 1
            
            if show_animation:
                plt.cla()
                if path:
                    plt.plot(rx, ry, "-r", linewidth=2)
                plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
                plt.plot(x[0], x[1], "xr")
                plt.plot(goal[0], goal[1], "xb")
                self.plot_obstacles(current_obstacles)
                
                plot_arrow(x[0], x[1], x[2])
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.0001)
            
            # goal is reached
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

    def plot_obstacles(self, obstacles):
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], fill=True)
            plt.gca().add_artist(circle)

def main():
    print("AStarDWAAgent simulation start")
    
    map_size = 50.0
    
    sx, sy = 5.0, 5.0
    gx, gy = 45.0, 45.0
    
    grid_size = 2.0
    robot_radius = 1.0

    static_obstacles = [
        (10.0, 10.0, 1.0),
        (20.0, 20.0, 1.0),
        (30.0, 26.0, 1.0),
        (40.0, 42.0, 1.0),
        (15.0, 35.0, 1.0),
        (35.0, 15.0, 1.0)
    ]

    dynamic_obstacles = []
    
    print(f"Start position: ({sx}, {sy})")
    print(f"Goal position: ({gx}, {gy})")
    print(f"Static obstacles: {static_obstacles}")
    print(f"Dynamic obstacles: {dynamic_obstacles}")

    agent = AStarDWAAgent(static_obstacles, grid_size, robot_radius)

    path = agent.plan_path(sx, sy, gx, gy)
    
    trajectory = agent.move_to_goal([sx, sy], [gx, gy], path, dynamic_obstacles)
    
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()

if __name__ == "__main__":
    main()