# path_planner.py（修正版）
import heapq
import numpy as np
from shapely.geometry import Point, LineString  # 添加Point的显式导入
from discrete_graph import is_valid_connection

class PathPlanner:
    def __init__(self, graph, game_map):
        self.graph = graph
        self.game_map = game_map
        self.start = None  # 类型为shapely.geometry.Point
        self.goal = None   # 类型为shapely.geometry.Point

    def dijkstra_path(self, start: Point, goal: Point):
        """改进后的Dijkstra实现（包含输入验证和鲁棒性增强）"""
        # 输入验证
        if start not in self.graph:
            raise ValueError("起点不在离散图中")
        if goal not in self.graph:
            raise ValueError("终点不在离散图中")

        self.start = start
        self.goal = goal

        def node_key(point):
            return round(point.x, 6), round(point.y, 6)  # 提高精度

        frontier = []
        start_key = node_key(start)
        heapq.heappush(frontier, (0, start_key[0], start_key[1], hash(start), start))

        came_from = {}
        cost_so_far = {start_key: 0}

        while frontier:
            current_cost, _, _, _, current_node = heapq.heappop(frontier)
            current_key = node_key(current_node)

            if current_node == goal:
                break

            for neighbor, edge_cost in self.graph.get(current_node, []):
                neighbor_key = node_key(neighbor)
                new_cost = cost_so_far[current_key] + edge_cost

                if neighbor_key not in cost_so_far or new_cost < cost_so_far[neighbor_key]:
                    cost_so_far[neighbor_key] = new_cost
                    heapq.heappush(
                        frontier,
                        (new_cost, neighbor_key[0], neighbor_key[1], hash(neighbor), neighbor)
                    )
                    came_from[neighbor_key] = current_key

        # 路径回溯（增强鲁棒性）
        path = []
        current = node_key(goal)
        start_key = node_key(start)

        try:
            while current != start_key:
                point = next(p for p in self.graph if node_key(p) == current)
                path.append(point)
                current = came_from.get(current)
                if current is None:
                    return []
            path.append(start)
            path.reverse()
        except StopIteration:
            return []

        return path


    @staticmethod
    def visualize_paths(original, game_map):
        """可视化路径对比"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(game_map, cmap='gray_r', origin='lower')

        # 绘制原始路径
        ox = [p.x for p in original]
        oy = [p.y for p in original]
        plt.plot(ox, oy, 'r--', label='Original Path', linewidth=2)

        plt.scatter(ox, oy, c='red', s=30, zorder=3)
        plt.legend()
        plt.title("Path Optimization Comparison")
        plt.show()

