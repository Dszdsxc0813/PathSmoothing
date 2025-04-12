# path_corrector.py
import math
import numpy as np
from shapely.geometry import Point, LineString


class PathCorrector:
    def __init__(self, turn_radius=4.0):
        """
        路径校正核心计算模块
        :param turn_radius: 移动机器人最小转弯半径
        """
        self.R = turn_radius
        self.epsilon = 1e-6  # 浮点计算容差

    def calculate_correction(self, current_pos, segment_start, segment_end, next_segment_end, alpha_deg, beta_deg):
        """
        执行完整路径校正计算
        :param current_pos: 当前位置 (x0, y0)
        :param segment_start: 当前路径段起点 (x1, y1)
        :param segment_end: 当前路径段终点 (x2, y2)
        :param alpha_deg: 机器人当前朝向角度（度）
        :return: 校正路径点列表
        """
        # 计算O2圆心（公式6-7）
        x0, y0 = current_pos
        x1, y1 = segment_start
        x2, y2 = segment_end
        alpha = math.radians(alpha_deg)

        # 公式6-7
        x_O2 = x0 - self.R * math.sin(alpha - math.pi / 2)
        y_O2 = y0 - self.R * math.cos(alpha - math.pi / 2)

        # 构造直线L方程（公式8）
        k = self._calculate_slope(segment_start, segment_end)
        if k is None:  # 垂直线处理
            L_a = 1
            L_b = 0
            L_c = -(x1 + self.R * math.sin(alpha))
        else:
            L_a = k
            L_b = -1
            offset = self.R * math.sqrt(k ** 2 + 1)
            L_c = -k * (x1 + self.R * math.sin(alpha)) + (y1 + self.R * math.cos(alpha)) - offset

        # 构造圆O2方程（公式9）
        circle_center = (x_O2, y_O2)
        circle_radius = 2 * self.R

        # 求解直线L与圆O2的交点（公式10-14）
        intersections = self._circle_line_intersection(
            circle_center, circle_radius,
            (L_a, L_b, L_c)
        )

        if not intersections:
            return []  # 无交点情况

        # 选择有效交点（公式15）
        x_O3, y_O3 = self._select_valid_intersection(
            intersections, segment_start, segment_end, k
        )

        # 计算矫正终点X（公式17-18）
        x_X = x_O3 + self.R * math.sin(alpha)
        y_X = y_O3 - self.R * math.cos(alpha)
        correction_point = Point(x_X, y_X)

        # 计算距离参数（公式19-20）
        d1 = self._calculate_distance((x_X, y_X), segment_start)
        d2 = self._calculate_distance(next_segment_end, segment_start)

        # 动作决策（论文第3节末尾）
        action = self._determine_action(beta_deg, d1, d2)

        # 生成修正路径点
        return self._generate_action_path(
            action=action,
            current_pos=current_pos,
            target_point=(x_X, y_X),
            segment_start=segment_start,
            segment_end=segment_end,
            next_segment_end=next_segment_end
        )

    def _calculate_slope(self, start, end):
        """计算路径段斜率（公式12）"""
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < self.epsilon:
            return None  # 垂直线
        return dy / dx

    def _circle_line_intersection(self, center, radius, line_coeff):
        """
        圆与直线交点计算（公式10-14）
        :param center: 圆心 (h, k)
        :param radius: 圆半径
        :param line_coeff: 直线系数 (A, B, C) 对应Ax + By + C = 0
        :return: 交点列表
        """
        A, B, C = line_coeff
        h, k = center
        r = radius

        # 将直线方程转换为标准形式
        numerator = A * h + B * k + C
        denom = A ** 2 + B ** 2

        if denom < self.epsilon:
            return []

        # 计算判别式
        d = r ** 2 * denom - (A * h + B * k + C) ** 2

        if d < 0:
            return []

        # 计算交点坐标
        sqrt_d = math.sqrt(d)
        x_base = (B * (B * h - A * k) - A * C) / denom
        y_base = (A * (-B * h + A * k) - B * C) / denom

        dx = B * sqrt_d / denom
        dy = A * sqrt_d / denom

        return [
            (x_base + dx, y_base - dy),
            (x_base - dx, y_base + dy)
        ]

    def _select_valid_intersection(self, points, start, end, slope):
        """
        选择有效交点（公式15）
        :param points: 候选交点列表
        :param start: 路径起点 (x1, y1)
        :param end: 路径终点 (x2, y2)
        :param slope: 路径斜率
        :return: 选择的交点坐标
        """
        x1, y1 = start
        x2, y2 = end
        valid_points = []

        for px, py in points:
            # 严格实现论文公式15条件
            denominator = x2 - x1
            if abs(denominator) < self.epsilon:
                denominator = self.epsilon

            ratio = (px - points[1][0]) / denominator  # (xa - xb)/(x2 - x1)

            if ratio > 0:
                valid_points.append((px, py))
            else:
                valid_points.append(points[1])  # 选择另一个交点

        # 选择最接近路径方向的点
        return self._select_by_direction(valid_points, start, end)

    def _is_valid_correction(self, start, end, segment_start, segment_end):
        """
        验证校正路径可行性
        :param start: 校正起点
        :param end: 校正终点
        :param segment_start: 原路径起点
        :param segment_end: 原路径终点
        """
        # 创建校正路径线段
        correction_line = LineString([start, end])

        # 创建原路径线段
        original_segment = LineString([segment_start, segment_end])

        # 检查两线段是否相交（至少接触）
        return correction_line.intersects(original_segment)

    def _select_by_direction(self, points, start, end):
        """根据路径方向选择最优交点"""
        path_vector = np.array(end) - np.array(start)
        max_dot = -float('inf')
        best_point = points[0]

        for p in points:
            point_vector = np.array(p) - np.array(start)
            dot_product = np.dot(path_vector, point_vector)
            if dot_product > max_dot:
                max_dot = dot_product
                best_point = p
        return best_point

    def _calculate_distance(self, p1, p2):
        """计算两点间距离（公式19-20基础）"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _determine_action(self, beta, d1, d2):
        """动作决策逻辑（论文第3节）"""
        if beta > 0:
            if d1 > d2:
                return "left_reverse"
            else:
                return "right_forward"
        else:
            if d1 > d2:
                return "right_reverse"
            else:
                return "left_forward"

    def _generate_action_path(self, action, current_pos, target_point,
                              segment_start, segment_end, next_segment_end):
        """根据动作生成路径点"""
        path_points = []

        # 公共参数
        x0, y0 = current_pos
        x_X, y_X = target_point
        xn, yn = next_segment_end

        # 生成转向圆弧
        if "left" in action:
            arc = self._generate_arc(current_pos, target_point, clockwise=False)
        else:
            arc = self._generate_arc(current_pos, target_point, clockwise=True)

        path_points.extend(arc)

        # 添加倒车路径（如果需要）
        if "reverse" in action:
            back_path = self._generate_reverse_path(
                start_point=target_point,
                end_point=segment_end,
                safe_distance=2 * self.R
            )
            path_points.extend(back_path)

        # 连接下一段路径
        transition = LineString([target_point, next_segment_end])
        path_points.extend(transition.coords)

        return [Point(p) for p in path_points]

    def _generate_arc(self, start, end, clockwise=True, num_points=8):
        """生成转向圆弧点集"""
        center = self._calculate_arc_center(start, end, clockwise)
        radius = self.R

        # 计算起止角度
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])

        # 调整角度方向
        if clockwise:
            if end_angle > start_angle:
                end_angle -= 2 * math.pi
        else:
            if end_angle < start_angle:
                end_angle += 2 * math.pi

        # 生成点集
        angles = np.linspace(start_angle, end_angle, num_points)
        return [
            Point(
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle)
            ) for angle in angles
        ]

    def _calculate_arc_center(self, start, end, clockwise):
        """计算圆弧圆心（根据转向方向）"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        direction = 1 if clockwise else -1
        return (
            start[0] - direction * dy,
            start[1] + direction * dx
        )

    def _generate_reverse_path(self, start_point, end_point, safe_distance):
        """生成倒车路径"""
        path_vector = np.array(end_point) - np.array(start_point)
        unit_vector = path_vector / np.linalg.norm(path_vector)

        back_step = unit_vector * safe_distance
        reverse_point = start_point - back_step

        return [Point(reverse_point), Point(start_point)]
