# path_smoother.py
import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from path_corrector import PathCorrector


class PathSmoother:
    def __init__(self, game_map, turn_radius=4.0):
        """
        增强版路径平滑优化器
        :param game_map: 二维障碍物矩阵（0=可行，1=障碍）
        :param turn_radius: 最小转弯半径（像素单位）
        """
        self.game_map = game_map
        self.turn_radius = turn_radius
        self.safety_margin = 2.0  # 安全边界距离
        self.corrector = PathCorrector(self.turn_radius+self.safety_margin)

    def smooth_path(self, raw_path):
        """
        增强优化流程
        :param raw_path: 引导路径（Point对象列表）
        :return: 几何拓扑优化后的路径（Point对象列表）
        """
        if len(raw_path) < 3:
            return raw_path

        # 第二阶段：几何拓扑优化
        optimized_path = self._geometric_optimization(raw_path)

        return optimized_path

    def _geometric_optimization(self, path):
        """几何拓扑优化核心方法"""
        optimized = []
        path = [Point(p) for p in path]  # 确保转为Point对象

        for i in range(len(path) - 1):
            current_point = path[i]
            next_point = path[i + 1]

            # 添加当前点
            optimized.append(current_point)

            # 获取路径段信息
            if i == 0:
                prev_point = current_point  # 第一个点特殊处理
            else:
                prev_point = path[i - 1]

            # 计算当前朝向（假设为前一节点的移动方向）
            dx = current_point.x - prev_point.x
            dy = current_point.y - prev_point.y
            alpha_deg = math.degrees(math.atan2(dy, dx)) if (dx, dy) != (0, 0) else 0

            # 计算几何参数
            beta = self._calculate_beta(
                alpha_deg=alpha_deg,
                current_pos=(current_point.x, current_point.y),
                segment_start=(prev_point.x, prev_point.y),
                segment_end=(next_point.x, next_point.y)
            )

            D_ac_expected = self._calculate_expected_distance(beta)
            D_ac_actual = self._calculate_actual_distance(
                current_pos=(current_point.x, current_point.y),
                segment_start=(prev_point.x, prev_point.y),
                segment_end=(next_point.x, next_point.y)
            )

            # 执行路径修正
            corrected_points = self._apply_correction(
                current_point=current_point,
                next_point=next_point,
                beta=beta,
                D_ac_expected=D_ac_expected,
                D_ac_actual=D_ac_actual
            )

            # 添加修正点
            optimized.extend(corrected_points)

        optimized.append(path[-1])
        return self._remove_collision_points(optimized)

    def _calculate_beta(self, alpha_deg, current_pos, segment_start, segment_end):
        """公式3实现：计算前进方向与目标路径夹角β"""
        x0, y0 = current_pos
        x1, y1 = segment_start
        x2, y2 = segment_end

        # 计算路径方向角
        dx_segment = x2 - x1
        dy_segment = y2 - y1
        theta_rad = math.atan2(dy_segment, dx_segment)

        # 计算β
        alpha_rad = math.radians(alpha_deg)
        beta_rad = theta_rad - alpha_rad
        beta_deg = math.degrees(beta_rad)

        # 规范化到[-180, 180]
        beta_deg = (beta_deg + 180) % 360 - 180
        return beta_deg

    def _calculate_expected_distance(self, beta):
        """公式4实现：计算预期距离"""
        beta_rad = math.radians(beta)
        return 2 * self.turn_radius * (math.sin(beta_rad / 2) ** 2)

    def _calculate_actual_distance(self, current_pos, segment_start, segment_end):
        """公式5实现：计算实际距离"""
        x0, y0 = current_pos
        x1, y1 = segment_start
        x2, y2 = segment_end

        numerator = abs(
            (x1 * y2 - x1 * y0) +
            (x2 * y0 - x2 * y1) +
            (x0 * y1 - x0 * y2)
        )
        denominator = math.hypot(x2 - x1, y2 - y1)

        return numerator / denominator if denominator > 1e-6 else float('inf')

    def _apply_correction(self, current_point, next_point, beta, D_ac_expected, D_ac_actual):
        """应用路径校正逻辑"""
        correction_points = []

        # 状态判断
        if D_ac_actual > D_ac_expected + self.safety_margin:
            # 保持当前路径
            return []

        elif abs(D_ac_actual - D_ac_expected) <= self.safety_margin:
            # 生成圆弧过渡
            arc_points = self._generate_turn_arc(
                current_point=current_point,
                next_point=next_point,
                beta=beta
            )
            correction_points.extend(arc_points)

        else:
            # 需要路径矫正
            corrected = self._calculate_correction_path(
                current_point=current_point,
                next_point=next_point,
                beta=beta
            )
            correction_points.extend(corrected)

        return correction_points

    def _generate_turn_arc(self, current_point, next_point, beta, num_points=10):
        """生成转弯圆弧点集"""
        radius = self.turn_radius
        center = self._calculate_arc_center(current_point, beta, radius)

        # 计算起始和终止角度
        start_angle = math.atan2(current_point.y - center.y,
                                 current_point.x - center.x)
        end_angle = start_angle + math.radians(beta)

        # 生成圆弧点
        points = []
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append(Point(x, y))

        return points

    def _calculate_arc_center(self, current_point, beta, radius):
        """计算圆弧圆心坐标"""
        beta_rad = math.radians(beta)
        dx = radius * math.cos(beta_rad / 2)
        dy = radius * math.sin(beta_rad / 2)
        return Point(current_point.x - dx, current_point.y + dy)

    def _remove_collision_points(self, path):
        """移除导致碰撞的路径点"""
        safe_path = []
        for point in path:
            if not self._is_collision(point):
                safe_path.append(point)
        return safe_path

    def _is_collision(self, point):
        """碰撞检测"""
        x, y = int(point.x), int(point.y)
        if 0 <= x < self.game_map.shape[1] and 0 <= y < self.game_map.shape[0]:
            return self.game_map[y][x] == 1
        return True  # 越界视为碰撞

    def _calculate_correction_path(self, current_point, next_point, beta):
        # 获取必要参数
        current_pos = (current_point.x, current_point.y)
        segment_start = ...  # 从前驱节点获取
        segment_end = (next_point.x, next_point.y)
        alpha_deg = ...  # 从机器人状态获取

        # 调用校正模块
        return self.corrector.calculate_correction(
            current_pos, segment_start, segment_end, alpha_deg
        )

