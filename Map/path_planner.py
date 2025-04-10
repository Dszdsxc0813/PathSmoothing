# path_planner.py
import heapq
import math
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

    def optimize_path(self, original_path):
        """路径优化函数"""
        if len(original_path) < 3:
            return original_path.copy()

        optimized = list(original_path)
        modified = True

        while modified:
            modified = False
            i = 1  # 从第二个节点开始检查

            while i < len(optimized) - 1:
                prev_node = optimized[i - 1]
                curr_node = optimized[i]
                next_node = optimized[i + 1]

                # 检查前后节点是否可直接连接
                if self.is_valid_connection(prev_node, next_node):
                    del optimized[i]
                    modified = True
                else:
                    i += 1
        return optimized

    def is_valid_connection(self, node1, node2):
        """碰撞检测核心方法"""
        line = LineString([(node1.x, node1.y), (node2.x, node2.y)])

        # 获取线段经过的所有网格坐标
        x0, y0 = node1.x, node1.y
        x1, y1 = node2.x, node2.y

        dx = x1 - x0
        dy = y1 - y0
        distance = math.hypot(dx, dy)

        # 采样间隔设为0.5个地图单位
        steps = max(int(distance / 0.5), 1)
        step_x = dx / steps
        step_y = dy / steps

        for i in range(steps + 1):
            x = x0 + i * step_x
            y = y0 + i * step_y
            if self.is_obstacle(x, y):
                return False
        return True


    def is_obstacle(self, x, y):
        """判断指定坐标是否为障碍物"""
        map_h, map_w = self.game_map.shape
        if x < 0 or x >= map_w or y < 0 or y >= map_h:
            return True  # 超出边界视为障碍物
        return self.game_map[int(y)][int(x)] == 1


    @staticmethod
    def visualize_paths(original,optimized, game_map):
        """可视化路径对比"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(game_map, cmap='gray_r', origin='lower')

        # 绘制原始路径
        ox = [p.x for p in original]
        oy = [p.y for p in original]
        plt.plot(ox, oy, 'r--', label='Original Path', linewidth=2, alpha=0.7)
        plt.scatter(ox, oy, c='red', s=30, zorder=3)

        # 绘制优化路径
        if optimized:
            opt_x = [p.x for p in optimized]
            opt_y = [p.y for p in optimized]
            plt.plot(opt_x, opt_y, 'b-', label='Optimized Path', linewidth=2)
            plt.scatter(opt_x, opt_y, c='blue', s=50, zorder=3, marker='s')

        plt.legend()
        plt.title("Path Optimization Comparison")
        plt.show()

