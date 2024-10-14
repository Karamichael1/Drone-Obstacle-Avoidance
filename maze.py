import numpy as np
import matplotlib.pyplot as plt

class MazeGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.start = (1, 1)
        self.goal = (width - 3, height - 3)
        self.mazes = self._generate_mazes()

    def _generate_mazes(self):
        mazes = []
        for i in range(1, 11):
            obstacle_count = i * 10  
            maze = self._create_maze(obstacle_count)
            mazes.append(maze)
        return mazes

    def _create_maze(self, obstacle_count):
        maze = np.zeros((self.height, self.width))
        
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
        
        
        added_obstacles = 0
        while added_obstacles < obstacle_count:
            i = np.random.randint(1, self.height - 1)
            j = np.random.randint(1, self.width - 1)
            if maze[i, j] == 0 and (j, i) != self.start and (j, i) != self.goal:
                maze[i, j] = 1
                added_obstacles += 1
        
        
        maze[self.start[1], self.start[0]] = 0
        maze[self.goal[1], self.goal[0]] = 0
    
        return maze

    def get_maze(self, difficulty):
        if 1 <= difficulty <= 10:
            return self.mazes[difficulty - 1]
        else:
            raise ValueError("Difficulty must be between 1 and 10")

    @staticmethod
    def maze_to_obstacles(maze):
        obstacles = []
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] == 1:
                    obstacles.append((j, i, 0.5))
        return obstacles

if __name__ == "__main__":
    maze_gen = MazeGenerator(50, 50)
    
    for i in range(1, 11):
        maze = maze_gen.get_maze(i)
        obstacle_count = np.sum(maze) - (2 * maze.shape[0] + 2 * maze.shape[1] - 4)  
        print(f"Maze {i}: Obstacle count = {obstacle_count}")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        maze = maze_gen.get_maze(i + 1)
        ax.imshow(maze, cmap='binary')
        ax.plot(maze_gen.start[0], maze_gen.start[1], 'go', markersize=8, label='Start')
        ax.plot(maze_gen.goal[0], maze_gen.goal[1], 'ro', markersize=8, label='Goal')
        ax.set_title(f"Maze {i + 1}")
        ax.axis('off')
        if i == 0:  
            ax.legend()
    
    plt.tight_layout()
    plt.show()