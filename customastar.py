import numpy as np
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

        obstacle_set = set((round(obs[0] / self.grid_size), round(obs[1] / self.grid_size))
                           for obs in obstacles)

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

        return None  # No path found

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