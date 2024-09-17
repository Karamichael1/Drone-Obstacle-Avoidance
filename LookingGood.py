import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from dynamicwindow2D import Config, RobotType, dwa_control, motion, plot_arrow

show_animation = True

class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_index)

class AStarPlanner:
    def __init__(self, ox, oy, resolution, robot_radius):
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    def planning(self, sx, sy, gx, gy):
        start_node = Node(self.calc_xy_index(sx, self.min_x),
                          self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = Node(self.calc_xy_index(gx, self.min_x),
                         self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = Node(current.x + self.motion[i][0],
                            current.y + self.motion[i][1],
                            current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * (abs(n1.x - n2.x) + abs(n1.y - n2.y))  # Manhattan distance
        return d

    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        return [[1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
                [-1, -1, math.sqrt(2)],
                [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)],
                [1, 1, math.sqrt(2)]]

class AStarDWAAgent:
    def __init__(self, static_obstacles, resolution, robot_radius,max_speed):
        self.static_obstacles = static_obstacles
        ox = [obs[0] for obs in self.static_obstacles]
        oy = [obs[1] for obs in self.static_obstacles]
        self.a_star = AStarPlanner(ox, oy, resolution, robot_radius)
        self.dwa_config = Config()
        self.dwa_config.robot_type = RobotType.circle
        self.dwa_config.robot_radius = robot_radius
        self.dwa_config.max_speed = max_speed
        self.collision_check_distance = robot_radius * 0.5

    def plan_path(self, sx, sy, gx, gy):
        return self.a_star.planning(sx, sy, gx, gy)

    def move_to_goal(self, start, goal, path, dynamic_obstacles):
        x = np.array(start + [math.pi / 8.0, 0.0, 0.0])
        trajectory = np.array(x)
        
        rx, ry = path
        target_ind = 0  
        
        time = 0
        while True:
            current_obstacles = self.update_obstacles(time, dynamic_obstacles)
            dwa_obstacles = np.array([[obs[0], obs[1]] for obs in current_obstacles])
            
            if target_ind < len(rx):
                local_goal = [rx[target_ind], ry[target_ind]]
            else:
                local_goal = goal
            
            if self.check_collision(x[:2], current_obstacles):
                print("Potential collision detected. Using DWA for avoidance.")
                u, predicted_trajectory = dwa_control(x, self.dwa_config, local_goal, dwa_obstacles)
            else:
                angle = math.atan2(local_goal[1] - x[1], local_goal[0] - x[0])
                speed = min(self.dwa_config.max_speed, math.hypot(local_goal[0] - x[0], local_goal[1] - x[1]))
                u = np.array([speed, angle - x[2]])
                predicted_trajectory = np.array([x[:2], local_goal])
            
            x = motion(x, u, self.dwa_config.dt)
            trajectory = np.vstack((trajectory, x))
            
            dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
            if dist_to_goal <= 0.5:  # Check if within 0.5 units of the goal
                print("Goal reached!")
                break
            
            dist_to_target = math.hypot(x[0] - local_goal[0], x[1] - local_goal[1])
            if dist_to_target <= self.dwa_config.robot_radius:
                if target_ind < len(rx) - 1:
                    target_ind += 1
            
            if show_animation:
                self.plot_state(x, goal, rx, ry, predicted_trajectory, current_obstacles)
            
            time += self.dwa_config.dt
        
        return trajectory

    def check_collision(self, position, obstacles):
        for obs in obstacles:
            if math.hypot(position[0] - obs[0], position[1] - obs[1]) <= self.collision_check_distance:
                return True
        return False

    def update_obstacles(self, time, dynamic_obstacles):
        current_obstacles = self.static_obstacles.copy()
        for obs in dynamic_obstacles:
            x = obs['x'] + obs['vx'] * time
            y = obs['y'] + obs['vy'] * time
            current_obstacles.append((x, y, obs['radius']))  
        return current_obstacles

    def plot_state(self, x, goal, rx, ry, predicted_trajectory, obstacles):
        plt.cla()
        plt.plot(x[0], x[1], "xr")
        plt.plot(goal[0], goal[1], "xb")
        
        self.plot_obstacles(obstacles)
        
        plot_arrow(x[0], x[1], x[2])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

    def plot_obstacles(self, obstacles):
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], fill=True)
            plt.gca().add_artist(circle)

def create_maze(width, height, obstacle_ratio):  
    maze = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height-1 or j == 0 or j == width-1:
                maze[i, j] = 1 
            elif np.random.rand() < obstacle_ratio:
                maze[i, j] = 1 
    return maze

def maze_to_obstacles(maze):
    obstacles = []
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 1:
                obstacles.append((j, i, 0.5)) 
    return obstacles

def main():
    print("AStarDWAAgent simulation start")
    
    maze_width, maze_height = 50, 50
    maze = create_maze(maze_width, maze_height, obstacle_ratio=0.01) 
    
    sx, sy = 1.0, 1.0
    gx, gy = maze_width - 3.0, maze_height - 3.0
    
    grid_size = 1.0
    robot_radius = 0.5

    static_obstacles = maze_to_obstacles(maze)

    dynamic_obstacles = []
    
    print(f"Start position: ({sx}, {sy})")
    print(f"Goal position: ({gx}, {gy})")

    agent = AStarDWAAgent(static_obstacles, grid_size, robot_radius, max_speed=8.0)
    path = agent.plan_path(sx, sy, gx, gy)
    if path:
        trajectory = agent.move_to_goal([sx, sy], [gx, gy], path, dynamic_obstacles)
        
        if show_animation:
            plt.figure(figsize=(10, 10))
            plt.imshow(maze, cmap='binary')
            plt.plot(trajectory[:, 0], trajectory[:, 1], "-r", linewidth=2)
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.title("AStarDWAAgent in Maze")
            plt.axis("equal")
            plt.show()
    else:
        print("No path found")


if __name__ == "__main__":
    main()