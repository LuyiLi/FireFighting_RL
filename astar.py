import math
import heapq


class AStar():
    def heuristic_cost_estimate(self, current, target):
        # Calculate the distance as a heuristic
        dy = abs(current[1] - target[1])
        dx = abs(current[0] - target[0])
        return dy + dx + (1.414 - 2) * min(dx, dy)


    def shortest_path(self, grid, initial, target):
        # Define possible moves
        # moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Create a priority queue (min heap) for open nodes
        open_nodes = []

        # Initialize the open list with the initial node
        heapq.heappush(open_nodes, (0, initial))

        # Create dictionaries to store the cost and parent of each node
        cost_so_far = {initial: 0}
        came_from = {}

        while open_nodes:
            # Get the node with the lowest cost from the priority queue
            _, current_node = heapq.heappop(open_nodes)
            current_cost = cost_so_far[current_node]

            # Check if we've reached the target node
            if current_node == target:
                # Reconstruct the path from the target to the initial node
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(initial)
                path.reverse()
                return path, current_cost

            # Explore the neighbors of the current node
            for move in moves:
                dx, dy = move
                neighbor = (current_node[0] + dx, current_node[1] + dy)

                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                    if grid[neighbor[0]][neighbor[1]] - grid[current_node[0]][current_node[1]] != 0:
                        dz = math.inf
                    else:
                        dz = 0

                    cost = current_cost + dz + (move[0] ** 2 + move[1] ** 2)**0.5

                    if neighbor not in cost_so_far or cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = cost
                        priority = cost + self.heuristic_cost_estimate(neighbor, target)
                        heapq.heappush(open_nodes, (priority, neighbor))
                        came_from[neighbor] = current_node

        return None, None


    def Run(self, map, start_pos, end_pos):
        data = map
        offset = [0, 0]
        start = start_pos
        end = end_pos
        start[0] = start[0] + offset[0]
        end[0] = end[0] + offset[0]
        start[1] += offset[1]
        end[1] += offset[1]

        shortest_path, cost = self.shortest_path(data, tuple(start), tuple(end))
        # print(shortest_path)
        # print(cost)
        return - start_pos[0] + shortest_path[1][0], - start_pos[1] + shortest_path[1][1]

