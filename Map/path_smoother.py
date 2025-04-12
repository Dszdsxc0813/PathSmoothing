# path_smoother.py
import numpy as np
from shapely.geometry import LineString, Point
from discrete_graph import is_valid_connection,precompute_obstacles

class PathSmoother:
    def __init__(self, game_map, turn_radius=32.0):
        """
        路径平滑优化器（修正障碍物预处理问题）
        :param game_map: 二维障碍物矩阵（0=可行，1=障碍）
        """
        self.game_map = game_map
        self.turn_radius = turn_radius
        self.obstacles = precompute_obstacles(game_map)  # 新增障碍物预处理

    def smooth_path(self, raw_path):
        """
        主优化函数：执行完整优化流程
        :param raw_path: 原始路径（Point对象列表）
        :return: 优化后的路径（Point对象列表）
        """
        if len(raw_path) < 3:
            return raw_path

        # 第一阶段：冗余节点删除
        simplified_path = self._remove_redundant_nodes(raw_path)

        # 第二阶段：几何拓扑优化
        optimized_path = self._geometric_optimization(simplified_path)

        return optimized_path

    def _remove_redundant_nodes(self, path):
        """
        冗余节点删除算法
        算法逻辑：
        1. 遍历路径中的每个中间节点
        2. 若前驱和后继节点可直接连接且不碰撞障碍物，则删除当前节点
        """
        optimized = path.copy()
        changed = True

        while changed:
            changed = False
            i = 1  # 从第二个节点开始检查

            while i < len(optimized) - 1:
                prev = optimized[i - 1]
                next_node = optimized[i + 1]

                # 检查直线连接是否可行
                # 修正参数顺序和内容
                if is_valid_connection(prev, next_node, self.game_map, self.obstacles):
                    del optimized[i]
                    changed = True
                else:
                    i += 1
            return optimized

    def _geometric_optimization(self, path):
        """
        几何拓扑优化
        算法流程：
        1. 计算每个路径段的转向角度
        2. 根据转弯半径生成圆弧过渡路径
        """
        if len(path) < 3:
            return path


        optimized = [path[0]]  # 起点必须保留

        for i in range(1, len(path) - 1):
            curr = path[i]
            prev = optimized[-1]
            next_node = path[i + 1]

            # 计算转向参数
            params = self._calculate_turn_parameters(prev, curr, next_node)

            if params['needs_correction']:
                # 生成矫正路径（论文图8实现）
                correction = self._generate_correction_path(params)
                optimized.extend(correction)
            else:
                optimized.append(curr)

        optimized.append(path[-1])  # 终点必须保留
        return optimized

    def _calculate_turn_parameters(self, prev, curr, next_node):
        """
        转向参数计算（论文公式3-5）
        返回：
        - beta：转向角度（度）
        - D_ac：理论转向距离
        - D_rc：实际转向距离
        """
        # 向量计算
        vec_prev = np.array([curr.x - prev.x, curr.y - prev.y])
        vec_next = np.array([next_node.x - curr.x, next_node.y - curr.y])

        # 计算夹角（公式3）
        angle_prev = np.arctan2(vec_prev[1], vec_prev[0])
        angle_next = np.arctan2(vec_next[1], vec_next[0])
        beta = np.degrees((angle_next - angle_prev + np.pi) % (2 * np.pi) - np.pi)

        # 理论转向距离（公式4）
        beta_rad = np.radians(abs(beta))
        D_ac = 2 * self.turn_radius * np.cos(np.pi / 2 - beta_rad / 2) * np.sin(beta_rad / 2)

        # 实际转向距离（公式5）
        # 使用点到直线距离公式
        a = vec_next[1]
        b = -vec_next[0]
        c = vec_next[0] * curr.y - vec_next[1] * curr.x
        numerator = abs(a * curr.x + b * curr.y + c)
        denominator = np.sqrt(a ** 2 + b ** 2)
        D_rc = numerator / denominator if denominator != 0 else 0

        return {
            'beta': beta,
            'D_ac': D_ac,
            'D_rc': D_rc,
            'needs_correction': D_rc < D_ac
        }

