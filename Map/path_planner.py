# path_planner.py
import heapq
import math
import matplotlib.pyplot as plt
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

        # 获取线段长度
        dx = x1 - x0
        dy = y1 - y0
        distance = math.hypot(dx, dy)

        # 采样间隔设为0.5个地图单位
        steps = max(int(distance / 0.5), 1)
        # 计算每一步在x和y方向上的增量step_x和step_y
        step_x = dx / steps
        step_y = dy / steps

        # 沿线段从起点到终点，每隔0.5个单位采样一个点
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
        # 检查地图上该位置是否为障碍物
        return self.game_map[int(y)][int(x)] == 1


    @staticmethod
    def visualize_paths(raw_path, optimized_path, game_map):
        """
        三路径对比可视化（支持路径缺失场景）
        :param raw_path:       原始路径（Point列表，允许为None）
        :param optimized_path: 引导路径（Point列表，允许为None）
        :param smooth_path:    平滑路径（Point列表，允许为None）
        :param game_map:       二维障碍物矩阵（np.ndarray）
        """
        # ========================== 画布初始化 ==========================
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体支持

        # ====================== 障碍物地图可视化 ========================
        # 使用灰度色图：0=白色（可行区域），1=黑色（障碍物）
        ax.imshow(game_map, cmap="binary",
                  origin="lower",  # 坐标系原点在左下角
                  extent=[0, game_map.shape[1], 0, game_map.shape[0]])

        # ====================== 路径数据预处理 ========================
        def get_path_coords(path):
            """安全提取路径坐标（处理空路径）"""
            return ([p.x for p in path], [p.y for p in path]) if path else (None, None)

        # 提取各路径坐标
        raw_x, raw_y = get_path_coords(raw_path)
        opt_x, opt_y = get_path_coords(optimized_path)

        # ====================== 路径可视化层 ========================
        # ---- 原始路径（红色虚线）----
        if raw_x and raw_y:
            ax.plot(raw_x, raw_y, 'r--',
                    linewidth=1.5, alpha=0.7,
                    label=f"原始路径 (节点数:{len(raw_path)})")
            # 绘制离散点，使用红色的 'x' 标注
            xs = [p.x for p in raw_path]
            ys = [p.y for p in raw_path]
            plt.scatter(xs, ys, s=20, c='red', marker='o')

        # ---- 引导路径（绿色点划线）----
        if opt_x and opt_y:
            ax.plot(opt_x, opt_y, 'g-.',
                    linewidth=2, markersize=6,
                    label=f"引导路径 (节点数:{len(optimized_path)})")
            # 绘制离散点，使用绿色的圆点标注
            xs = [p.x for p in optimized_path]
            ys = [p.y for p in optimized_path]
            plt.scatter(xs, ys, s=20, c='green', marker='x')

        # ==================== 起点终点标记层 ======================
        if raw_path:
            start_point = next((p for p in [raw_path, optimized_path] if p), None)
            goal_point = next((p[-1] for p in [raw_path, optimized_path] if p), None)
            if start_point:
                ax.scatter(start_point[0].x, start_point[0].y,
                           c='lime', s=200, marker='P', edgecolors='k', label="Start")
            if goal_point:
                ax.scatter(goal_point.x, goal_point.y,
                           c='gold', s=200, marker='*', edgecolors='k', label="Goal")

        # ====================== 坐标轴装饰 ========================
        ax.set_xlim(0, game_map.shape[1])
        ax.set_ylim(0, game_map.shape[0])
        ax.set_aspect('equal')  # 等比例坐标轴
        ax.grid(True, linestyle=':', color='gray', alpha=0.4)
        ax.set_xlabel("X 坐标", fontsize=12)
        ax.set_ylabel("Y 坐标", fontsize=12)
        ax.set_title("路径规划效果对比: 原始路径 → 引导路径",
                     fontsize=14, pad=15)

        # ====================== 图例与输出 ========================
        plt.tight_layout()
        plt.show()