import numpy as np
import math
from DStarLite import DStarLite, Node
from dynamicwindow2D import Config, dwa_control, RobotType

class HybridPlanner:
    def __init__(self, start, goal, static_obstacles, dynamic_obstacles):
        self.start = start
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.config = Config()
        
        # Calculate grid size based on start and goal positions
        max_x = max(start[0], goal[0], np.max(static_obstacles[:, 0]), np.max(dynamic_obstacles[:, 0])) + 10
        max_y = max(start[1], goal[1], np.max(static_obstacles[:, 1]), np.max(dynamic_obstacles[:, 1])) + 10
        min_x = min(start[0], goal[0], np.min(static_obstacles[:, 0]), np.min(dynamic_obstacles[:, 0])) - 10
        min_y = min(start[1], goal[1], np.min(static_obstacles[:, 1]), np.min(dynamic_obstacles[:, 1])) - 10
        
        # Initialize DStarLite with appropriate grid size
        self.dstar = DStarLite(static_obstacles[:, 0], static_obstacles[:, 1], x_min=min_x, y_min=min_y, x_max=max_x, y_max=max_y)
        self.current_path = None
        self.using_dwa = False

    def plan(self):
        # Initial D* Lite planning
        success, pathx, pathy = self.dstar.main(Node(x=self.start[0], y=self.start[1]),
                                                Node(x=self.goal[0], y=self.goal[1]),
                                                [], [])
        if not success:
            return None
        self.current_path = list(zip(pathx, pathy))
        
        # Main planning loop
        x = np.array([self.start[0], self.start[1], 0.0, 0.0, 0.0])  # x, y, yaw, v, omega
        trajectory = [x[:2]]
        
        while not self.goal_reached(x[:2]):
            if self.collision_imminent(x[:2]):
                # Switch to DWA
                self.using_dwa = True
                u, _ = dwa_control(x, self.config, self.goal, self.get_all_obstacles())
                x = self.motion(x, u, self.config.dt)
            else:
                # Follow D* Lite path
                self.using_dwa = False
                next_point = self.get_next_point(x[:2])
                if next_point is None:
                    # Replan with D* Lite
                    success, pathx, pathy = self.dstar.main(Node(x=x[0], y=x[1]),
                                                            Node(x=self.goal[0], y=self.goal[1]),
                                                            [], [])
                    if not success:
                        return None
                    self.current_path = list(zip(pathx, pathy))
                    next_point = self.get_next_point(x[:2])
                
                direction = math.atan2(next_point[1] - x[1], next_point[0] - x[0])
                u = [self.config.max_speed, self.angle_diff(direction, x[2])]
                x = self.motion(x, u, self.config.dt)
            
            trajectory.append(x[:2])
            self.update_dynamic_obstacles()
        
        return trajectory

    def collision_imminent(self, position):
        for obs in self.get_all_obstacles():
            if np.linalg.norm(position - obs) < self.config.robot_radius * 2:
                return True
        return False

    def get_all_obstacles(self):
        return np.vstack((self.static_obstacles, self.dynamic_obstacles))

    def get_next_point(self, position):
        if not self.current_path:
            return None
        for point in self.current_path:
            if np.linalg.norm(np.array(point) - position) > self.config.robot_radius:
                return point
        return None

    def goal_reached(self, position):
        return np.linalg.norm(position - self.goal) < self.config.robot_radius

    def update_dynamic_obstacles(self):
        # Simple random movement for dynamic obstacles
        for i in range(len(self.dynamic_obstacles)):
            self.dynamic_obstacles[i] += np.random.uniform(-0.5, 0.5, 2)

    @staticmethod
    def motion(x, u, dt):
        # Simple motion model
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    @staticmethod
    def angle_diff(a, b):
        d = a - b
        return (d + math.pi) % (2 * math.pi) - math.pi

def main():
    # Set up environment
    start = np.array([0, 0])
    goal = np.array([50, 50])
    static_obstacles = np.array([[20, 20], [30, 30], [40, 40]])
    dynamic_obstacles = np.array([[25, 25], [35, 35]])

    # Create and run planner
    planner = HybridPlanner(start, goal, static_obstacles, dynamic_obstacles)
    trajectory = planner.plan()

    if trajectory is None:
        print("No path found")
    else:
        print("Path found")
        # Visualize the result
        import matplotlib.pyplot as plt
        trajectory = np.array(trajectory)
        plt.figure(figsize=(10, 10))
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-b', label='Robot Path')
        plt.plot(start[0], start[1], 'go', label='Start')
        plt.plot(goal[0], goal[1], 'ro', label='Goal')
        plt.plot(static_obstacles[:, 0], static_obstacles[:, 1], 'ks', label='Static Obstacles')
        plt.plot(dynamic_obstacles[:, 0], dynamic_obstacles[:, 1], 'ms', label='Dynamic Obstacles')
        plt.legend()
        plt.grid(True)
        plt.title('Hybrid D* Lite and DWA Path Planning')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.show()

if __name__ == "__main__":
    main()