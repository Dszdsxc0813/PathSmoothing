# path_smoother.py
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from path_corrector import *
from robot_state import *
from utils import *


class PathSmoother:
    def __init__(self, game_map, turn_radius):
        """
        增强版路径平滑优化器
        :param game_map: 二维障碍物矩阵（0=可行，1=障碍）
        :param turn_radius: 最小转弯半径（像素单位）
        """
        self.game_map = game_map
        self.turn_radius = turn_radius
        self.safety_margin = 0  # 安全边界距离
        # 修正距离边界先不考虑，修正边界 = 最小障碍长度+安全边界距离
        self.corrector = PathCorrector(self.turn_radius+self.safety_margin)
        self.optimized_path = []  # 新增优化路径存储

    def smooth_path(self, guide_path, visualize_step):
        """
        主优化流程：模拟机器人沿路径动态移动并优化
        :param guide_path: 引导路径（Point对象列表）
        :return: 几何拓扑优化后的路径（Point对象列表）
        """
        # temp：先考虑多个点的优化
        if len(guide_path) < 2:
            return guide_path

        # 1. 初始化机器人状态：从起点开始，初始朝向由起点到第一个路径点
        # ！！这里还要替代成实际的飞机膨胀模型
        robot = RobotState(
            current_pos=Point(guide_path[0].x, guide_path[0].y),
            current_segment_idx=0,
            heading_angle=self._calculate_initial_heading(guide_path[0], guide_path[1]),
        )

        # 测试！！
        # 随机取第0路径段旁的一个点
        start_point = Point(79,50)
        next_point = Point(75,58)
        robot.current_pos = Point(start_point.x, start_point.y)
        robot.heading_angle = self._calculate_initial_heading(start_point, next_point)
        turn_direction = "left"

        optimized_path = [robot.current_pos]

        # 2. 遍历每一段引导路径段，进行动态修正
        while robot.current_segment_idx < len(guide_path) - 1:
            segment_start = guide_path[robot.current_segment_idx]
            segment_end = guide_path[robot.current_segment_idx + 1]
            if (robot.current_segment_idx + 2 < len(guide_path)):
                segment_next_end = guide_path[robot.current_segment_idx + 2]
            elif(robot.current_segment_idx + 1 < len(guide_path) ):
                print("机器人所在路径段为倒数第二个路径段剩最后两个点了")
                # 获取 AB 的方向向量
                dx = segment_end.x - segment_start.x
                dy = segment_end.y - segment_start.y
                length = math.hypot(dx, dy)
                if length == 0:
                    direction = (0, 0)
                else:
                    direction = (dx / length, dy / length)

                # 沿 AB 方向前移距离 d 得到新的 segment_next_end
                new_x = segment_end.x + self.turn_radius * direction[0]
                new_y = segment_end.y + self.turn_radius * direction[1]
                segment_next_end = Point(new_x, new_y)
            else:
                break

            # 2.1 在路径段开始时，计算初始beta和期望偏差
            # 公式3：计算前进方向偏差角β
            initial_beta = self._calculate_beta(
                robot, segment_start, segment_end
            )
            # 固定D_ac_expected为初始值
            D_ac_expected = 2 * self.turn_radius * math.sin(initial_beta/2) ** 2

            # 2.2 在当前段上循环前进/转向/矫正
            while True:  # 单路径段处理循环
                # 公式4-5：计算预期和实际偏移距离
                D_ac_actual = self._calculate_actual_distance(robot.current_pos,
                                                              segment_start,
                                                              segment_end)

                # ===== 状态机决策 =====
                delta = D_ac_actual - D_ac_expected
                tolerance = 0.01  # 设置绝对容差阈值
                pass_N = False

                # Case 1: 如果未到达转向点，持续前进
                if delta > tolerance:
                    # Case1：继续沿当前方向前进
                    if math.sin(initial_beta) > 0:
                        turn_direction = "left"
                    elif math.sin(initial_beta) < 0:
                        turn_direction = "right"
                    else:
                        turn_direction = None  # 完全对齐，无需转向
                    next_point, circle_center = self._move_forward(robot, segment_start, segment_end, pass_N)
                    # —— 新增：step 太小，说明切点就在当前位置，切到下一段 CD ——
                    if next_point == robot.current_pos and circle_center is not None:
                        # 计算 CD 段的切点
                        CD_start = segment_start
                        CD_end = segment_next_end
                        try:
                            end_pt = self._tangent_point_on_segment(
                                circle_center, CD_start, CD_end, self.turn_radius
                            )
                        except ValueError:
                            # 无法求出切点时，直接直线连接到 CD 段的终点
                            fallback_line = self._interpolate_line(
                                robot.current_pos,
                                CD_end,
                                num_points=8  # 插值数量可调
                            )
                            for pt in fallback_line:
                                # 在 optimized_path.append(next_point) 前添加碰撞检测
                                if self.corrector.check_collision_segment([next_point], self.game_map):
                                    print("前进方向碰撞障碍物，触发重规划！")
                                optimized_path.append(pt)
                                robot = self._update_robot_state(robot, pt, guide_path)
                                if visualize_step:
                                    visualize_dynamic_paths(
                                        guide_path=guide_path,
                                        optimized_path=optimized_path,
                                        game_map=self.game_map
                                    )
                            robot.current_segment_idx += 1
                            break
                        else:
                            # 生成并行走半圆弧
                            arc_pts = self.generate_arc(
                                self.game_map,
                                guide_path,
                                optimized_path,
                                center=circle_center,
                                R=self.turn_radius,
                                start_pt=robot.current_pos,
                                end_pt=end_pt,
                                turn_direction=turn_direction,
                                num_points=16
                            )
                            for pt in arc_pts:
                                # 在 optimized_path.append(next_point) 前添加碰撞检测
                                if self.corrector.check_collision_segment([next_point], self.game_map):
                                    print("前进方向碰撞障碍物，触发重规划！")
                                optimized_path.append(pt)
                                robot = self._update_robot_state(robot, pt, guide_path)
                                if visualize_step:
                                    visualize_dynamic_paths(
                                        guide_path=guide_path,
                                        optimized_path=optimized_path,
                                        game_map=self.game_map
                                    )
                            # 切换到下一段 CD
                            robot.current_segment_idx += 1
                            break

                    # 在 optimized_path.append(next_point) 前添加碰撞检测
                    if self.corrector.check_collision_segment([next_point], self.game_map):
                        print("前进方向碰撞障碍物，触发重规划！")
                        # 不能直接退出，不然就变成死循环了
                        break  # 退出当前循环，触发局部重规划
                    optimized_path.append(next_point)
                    robot = self._update_robot_state(robot, next_point, guide_path)

                    if visualize_step:
                        visualize_dynamic_paths(
                            guide_path=guide_path,
                            optimized_path=optimized_path,
                            game_map=self.game_map
                        )

                    # 添加强制退出机制
                    if self._check_stuck(optimized_path):
                        robot.current_segment_idx += 1
                        break

                # 接下来处理转向或矫正
                # Case 2: 到达转向点，根据β正负转向
                # 先转向，再出发下一轮矫正
                elif -tolerance <= delta <= tolerance:
                    # ———— 1）决定转向方向
                    # β = θ−α 反映了“路径方向”在“机器人朝向”上的偏位：
                    # sin(β)>0 → 路径在左侧 → 左转（逆时针）
                    # sin(β)<0 → 路径在右侧 → 右转（顺时针）
                    # ———— 1）决定转向方向 ————
                    # β = θ−α ，反映“路径方向”相对于“机器人朝向”的偏移：
                    #   sin(β)>0 → 左侧 → 左转（逆时针）
                    #   sin(β)<0 → 右侧 → 右转（顺时针）
                    if math.sin(initial_beta) > 0:
                        turn_direction = "left"
                    elif math.sin(initial_beta) < 0:
                        turn_direction = "right"
                    else:
                        turn_direction = None  # 完全对齐，无需转向

                    next_point, circle_center = self._move_forward(robot, segment_start, segment_end, pass_N)

                    # ———— 2）获取转换圆弧
                    corrected_points, tangent_point_e = self._execute_turn(
                        robot, segment_start, segment_end,
                        turn_direction, circle_center
                    )
                    # ———— 3）在圆弧上前进，直到触发矫正条件
                    for i, point in enumerate(corrected_points):
                        # 更新机器人到当前路径点
                        # 在 optimized_path.append(next_point) 前添加碰撞检测
                        if self.corrector.check_collision_segment([next_point], self.game_map):
                            print("前进方向碰撞障碍物，触发重规划！")
                        optimized_path.append(point)
                        robot = self._update_robot_state(robot, point, guide_path)

                        # 实时计算最新状态参数
                        D_ac_actual = self._calculate_actual_distance(robot.current_pos,
                                                                      segment_start,
                                                                      segment_end)

                        # 动态可视化
                        if visualize_step:
                            visualize_dynamic_paths(
                                guide_path=guide_path,
                                optimized_path=optimized_path,
                                game_map=self.game_map
                            )

                        # 检测矫正条件，机器人沿矫正路径移动
                        if D_ac_actual - D_ac_expected < -tolerance:
                            # 执行矫正并跳出循环
                            corrected_points = self.corrector._apply_correction(
                                robot,
                                segment_start,
                                segment_end,
                                segment_next_end,
                                self.game_map,
                                guide_path,
                                optimized_path,
                                self.turn_radius,
                                visualize_steps = None
                            )
                            if optimized_path is not None:
                                optimized_path.extend(corrected_points)
                                robot = self._update_robot_state(robot, corrected_points[-2], guide_path)
                                robot = self._update_robot_state(robot, corrected_points[-1], guide_path)
                                if visualize_step:
                                    visualize_dynamic_paths(
                                        guide_path=guide_path,
                                        optimized_path=optimized_path,
                                        game_map=self.game_map
                                    )
                                break  # 跳出转向点处理循环
                            else:
                                print("路径矫正失败，optimized_path为空，请添加路径优化方法！")

                    # ——— 4) 切换到下一段 BC —— 更新段索引 ———
                    robot.current_segment_idx += 1

                    # 退出当前路径段循环
                    break

                # Case 3: 超过转向点，执行路径矫正
                else:
                    # Case3：需要路径矫正
                    corrected_points = self.corrector._apply_correction(
                        robot,
                        segment_start,
                        segment_end,
                        segment_next_end,
                        self.game_map,
                        guide_path,
                        optimized_path,
                        self.turn_radius,
                        visualize_steps=None
                    )
                    if optimized_path is not None:
                        optimized_path.extend(corrected_points)
                        robot = self._update_robot_state(robot, corrected_points[-2], guide_path)
                        robot = self._update_robot_state(robot, corrected_points[-1], guide_path)
                        # ——— 4) 切换到下一段 BC —— 更新段索引 ———
                        robot.current_segment_idx += 1
                        # 退出当前路径段循环
                        break
                    else:
                        print("路径矫正失败，optimized_path为空，请添加路径优化方法！")

                # ===== 增加循环退出条件 =====
                # 条件1：判断是否到达当前路径段终点
                if self._reached_segment_end(robot.current_pos, segment_end):
                    robot.current_segment_idx += 1
                    break

                # 条件2：防止无限循环（超过最大步数）
                if len(optimized_path) > 1000:
                    robot.current_segment_idx += 1
                    break

                if visualize_step:
                    visualize_dynamic_paths(
                        guide_path=guide_path,
                        optimized_path=optimized_path,
                        game_map=self.game_map
                    )
        return optimized_path

    def _calculate_initial_heading(self, start_point, next_point):
        """计算初始朝向角α"""
        dx = next_point.x - start_point.x
        dy = next_point.y - start_point.y
        return math.atan2(dy, dx)  # 相对于x轴正方向

    def _calculate_beta(self, robot, segment_start, segment_end):
        """公式3实现：计算前进方向与目标路径夹角β"""
        x1 = segment_start.x
        y1 = segment_start.y
        x2 = segment_end.x
        y2 = segment_end.y

        # 计算路径方向角
        dx_segment = x2 - x1
        dy_segment = y2 - y1
        # 计算路径方向角θ（相对于x轴正方向的弧度）
        theta_rad = math.atan2(dy_segment, dx_segment) if (dy_segment, dx_segment) != (0, 0) else 0.0  # 修正点：交换dx和dy顺序

        # β =  θ - α/a
        beta_rad = theta_rad - robot.heading_angle

        # 规范化到[-π, π]
        beta_rad = (beta_rad + math.pi) % (2 * math.pi) - math.pi
        return beta_rad


    def _calculate_actual_distance(self, current_pos, segment_start, segment_end):
        """
        计算机器人当前位置到路径 AB 的垂线距离（对无限直线），公式：
            distance = |(P−A) × (B−A)| / |B−A|
        其中 × 表示 2D 向量的“伪”叉乘 (u.x*v.y - u.y*v.x)
        """
        x0, y0 = current_pos.x, current_pos.y
        x1, y1 = segment_start.x, segment_start.y
        x2, y2 = segment_end.x, segment_end.y

        # 向量 AB 和 AP
        dx, dy = x2 - x1, y2 - y1
        ap_x, ap_y = x0 - x1, y0 - y1

        # 分母：|AB|
        denom = math.hypot(dx, dy)
        if denom < 1e-6:
            # AB 退化为点时，返回 P 到 A 的距离
            return math.hypot(ap_x, ap_y)

        # 叉积的绝对值 / |AB|
        return abs(ap_x * dy - ap_y * dx) / denom

    def _move_forward(self, robot, segment_start, segment_end, pass_N):
        """沿当前方向前进到预期转向点（与ZH和AB相切的圆切点）"""
        current_x = robot.current_pos.x
        current_y = robot.current_pos.y
        A = Point(current_x, current_y)  # 当前机器人位置
        B = self._intersect_heading_with_path(robot, segment_start, segment_end)  # 交点 K
        C = segment_end  # 路径终点B

        try:
            N, center = self.corrector.compute_blend_start(A, B, C, self.turn_radius, pass_N)
        except ValueError:
            # 无法计算圆心或切点，返回简单前进
            step_size = self.turn_radius * 0.1
            dx_zh = math.cos(robot.heading_angle)
            dy_zh = math.sin(robot.heading_angle)
            return Point(
                current_x + dx_zh * step_size,
                current_y + dy_zh * step_size
            ), None

            # —— 新增的“极小步长”判断 ——
        dx_to_N = N.x - current_x
        dy_to_N = N.y - current_y
        raw_step = math.hypot(dx_to_N, dy_to_N)
        if raw_step < 1e-6:
            # 切点几乎就是当前位置，交给 smoother 自己去从这个圆心算到 CD 段的圆弧
            return Point(current_x, current_y), center

        # --- 计算从当前位置到切点的前进步长 ---
        dx = N.x - current_x
        dy = N.y - current_y
        step_size = math.hypot(dx, dy)
        step_size = min(step_size, self.turn_radius * 0.1)

        dx_zh = math.cos(robot.heading_angle)
        dy_zh = math.sin(robot.heading_angle)

        return (
            Point(current_x + dx_zh * step_size, current_y + dy_zh * step_size),
            center
        )

    def _intersect_heading_with_path(self, robot, seg_start, seg_end):
        """
        计算以机器人当前位置为起点、以朝向为方向的射线（ZH），
        与路径段 AB 的交点 K

        如果交点不在路径段 AB 上，则返回路径段 AB 的一个端点（通常是离机器人更近的端点）
        """
        x0, y0 = robot.current_pos.x, robot.current_pos.y
        dx1 = math.cos(robot.heading_angle)
        dy1 = math.sin(robot.heading_angle)

        x1, y1 = seg_start.x, seg_start.y
        x2, y2 = seg_end.x, seg_end.y
        dx2 = x2 - x1
        dy2 = y2 - y1

        # 解方程: (x0 + t1 * dx1, y0 + t1 * dy1) = (x1 + t2 * dx2, y1 + t2 * dy2)
        # 系数矩阵的行列式
        denom = dx1 * dy2 - dy1 * dx2

        if abs(denom) < 1e-9:
            # 两条直线几乎平行，返回路径段的一个端点（通常是离机器人更近的端点）
            dist_start = math.hypot(x1 - x0, y1 - y0)
            dist_end = math.hypot(x2 - x0, y2 - y0)
            if dist_start <= dist_end:
                return Point(x1, y1)
            else:
                return Point(x2, y2)

        # 计算参数 t1 和 t2
        t1 = ((x1 - x0) * dy2 - (y1 - y0) * dx2) / denom
        t2 = ((x0 + t1 * dx1 - x1) / dx2) if abs(dx2) > 1e-9 else ((y0 + t1 * dy1 - y1) / dy2)

        # 检查交点是否在路径段 AB 上（t2 ∈ [0, 1]）
        if 0 <= t2 <= 1:
            intersect_x = x0 + t1 * dx1
            intersect_y = y0 + t1 * dy1
            return Point(intersect_x, intersect_y)
        else:
            # 交点不在路径段 AB 上，返回路径段的一个端点（通常是离机器人更近的端点）
            dist_start = math.hypot(x1 - x0, y1 - y0)
            dist_end = math.hypot(x2 - x0, y2 - y0)
            if dist_start <= dist_end:
                return Point(x1, y1)
            else:
                return Point(x2, y2)

    def _check_stuck(self, path):
        """检测路径是否陷入死循环"""
        if len(path) < 10:
            return False
        # 检查最近5个点是否重复
        last_points = [(p.x, p.y) for p in path[-5:]]
        return len(set(last_points)) < 3

    def _execute_turn(self, robot, segment_start, segment_end, direction, circle_center):
        """Case 2: 执行转向，基于圆心生成到AB的切点圆弧"""
        # --- 计算目标路径段AB与圆的切点E ---
        # 获取线段AB的起点和终点坐标
        a = np.array([segment_start.x, segment_start.y])
        b = np.array([segment_end.x, segment_end.y])

        # 计算AB直线方程
        dx_ab = b[0] - a[0]
        dy_ab = b[1] - a[1]
        a_ab = -dy_ab
        b_ab = dx_ab
        c_ab = dy_ab * a[0] - dx_ab * a[1]
        norm_ab = np.hypot(a_ab, b_ab)

        # 计算圆到AB的切点E
        def calculate_tangent_point(center, a_line, b_line, c_line, norm_line, radius):
            x0, y0 = center
            denominator = a_line ** 2 + b_line ** 2
            px = (b_line * (b_line * x0 - a_line * y0) - a_line * c_line) / denominator
            py = (a_line * (-b_line * x0 + a_line * y0) - b_line * c_line) / denominator
            return (px, py)

        e_x, e_y = calculate_tangent_point(
            (circle_center.x, circle_center.y),
            a_ab, b_ab, c_ab, norm_ab, self.turn_radius
        )
        tangent_point_e = Point(e_x, e_y)

        # --- 计算当前点（机器人位置）和切点E相对于圆心的角度 ---
        current_pos = np.array([robot.current_pos.x, robot.current_pos.y])
        center_pos = np.array([circle_center.x, circle_center.y])

        # 计算起始角和终止角（弧度，范围[-π, π]）
        # 计算起点和终点角度（弧度）
        start_angle = np.arctan2(current_pos[1] - center_pos[1], current_pos[0] - center_pos[0])
        end_angle = np.arctan2(tangent_point_e.y - center_pos[1],tangent_point_e.x - center_pos[0])

        # --- 选择最短圆弧方向 ---
        # 计算原始角度差
        delta_theta = end_angle - start_angle

        """
        temp：强制转向方向与旋转方向匹配
        右转（顺时针）：强制 end_angle < start_angle，通过减少 end_angle 确保角度递减。
        左转（逆时针）：强制 end_angle > start_angle，通过增加 end_angle 确保角度递增
        """

        # 根据转向方向选择最短路径
        if direction == "right":
            # 右转应顺时针旋转（角度递减）
            if delta_theta > 0:
                end_angle -= 2 * np.pi
        else:
            # 左转应逆时针旋转（角度递增）
            if delta_theta < 0:
                end_angle += 2 * np.pi

        # --- 生成圆弧点（直接按调整后的角度插值）---
        theta_values = np.linspace(start_angle, end_angle, num=20)
        corrected_points = []
        for theta in theta_values:
            x = circle_center.x + self.turn_radius * np.cos(theta)
            y = circle_center.y + self.turn_radius * np.sin(theta)
            corrected_points.append(Point(x, y))

        return corrected_points, tangent_point_e

    def _update_robot_state(self, robot, new_pos, raw_path):
        """更新机器人状态（核心：动态更新前进方向）"""
        # 判断是否进入下一个路径段（基于剩余距离）
        if(robot.current_segment_idx + 1 < len(raw_path)):
            segment_end = raw_path[robot.current_segment_idx + 1]
            remaining_distance = math.hypot(
                new_pos.x - segment_end.x,
                new_pos.y - segment_end.y
            )
            # ！！
            # 这个路径切换的理解始终理解不了，感觉有问题
            # 如果机器人当前的路径段已经到下一阶段了，那
            if remaining_distance < self.turn_radius * 0.1:  # 更严格的切换阈值
                robot.current_segment_idx += 1

        # 更新朝向角（仅当移动距离非零时）
        dx = new_pos.x - robot.current_pos.x
        dy = new_pos.y - robot.current_pos.y
        if dx != 0 or dy != 0:
            new_heading = math.atan2(dy, dx)  # 相对于x轴
        else:
            new_heading = robot.heading_angle  # 保持原朝向

        return RobotState(
            current_pos=new_pos,
            current_segment_idx=robot.current_segment_idx,
            heading_angle=new_heading
        )


    def _reached_segment_end(self, current_pos, segment_end):
        """到达判断逻辑（增加容差系数）"""
        dx = segment_end.x - current_pos.x
        dy = segment_end.y - current_pos.y
        return math.hypot(dx, dy) < self.turn_radius * 0.05

    def _tangent_point_on_segment(self,
                                  center: Point,
                                  seg_start: Point,
                                  seg_end: Point,
                                  R: float) -> Point:
        """
        给定圆心 center 和路径段 seg_start→seg_end，求出那条路径上与圆相切、离 seg_start 最远的切点
        """
        # 单位切线方向
        dx = seg_end.x - seg_start.x
        dy = seg_end.y - seg_start.y
        L = math.hypot(dx, dy)
        if L < 1e-9:
            raise ValueError("Degenerate segment")
        ux, uy = dx / L, dy / L

        # w = seg_start - center
        wx = seg_start.x - center.x
        wy = seg_start.y - center.y

        # 求解 (w + s * u)·(w + s * u) = R^2
        # a = 1, b = 2·(w·u), c = (w·w - R^2)
        b = 2 * (wx * ux + wy * uy)
        c = wx*wx + wy*wy - R*R
        disc = b*b - 4*c
        eps = 1e-8
        if disc < -eps:
            raise ValueError("no tangent")
        disc = max(disc, 0.0)
        sd = math.sqrt(disc)
        s1 = (-b + sd) / 2
        s2 = (-b - sd) / 2

        cands = []
        for s in (s1, s2):
            if 0 <= s <= L or s < -eps :
                cands.append( Point(seg_start.x + ux*s,
                                    seg_start.y + uy*s) )
        if not cands:
            raise ValueError("tangent off segment")
        # 选离 seg_start 最远的那个
        return max(cands, key=lambda p: (p.x - seg_start.x)**2 + (p.y - seg_start.y)**2)


    def generate_arc(self, custom_map, guide_path, optimized_path, center: Point, R: float,
                     start_pt: Point, end_pt: Point,
                     turn_direction: str,
                     num_points: int = 16,):
        """
        生成圆弧点
        :param center: 圆心
        :param radius: 半径
        :param start_pt: 圆弧起点
        :param end_pt: 圆弧终点
        :param turn_direction: "left" 或 "right"
        :param num_points: 插值点数
        """
        # 1) 计算起始和结束的极角
        theta_start = math.atan2(start_pt.y - center.y, start_pt.x - center.x)
        theta_end = math.atan2(end_pt.y - center.y, end_pt.x - center.x)

        # 2) 按照转向方向，规范化 delta_theta 到最小弧度
        delta_theta = theta_end - theta_start
        if turn_direction == "left":
            # 想要 delta_theta ≥ 0，若 < 0 就加 2π
            if delta_theta < 0:
                delta_theta += 2 * math.pi
        elif turn_direction == "right":
            # 想要 delta_theta ≤ 0，若 > 0 就减 2π
            if delta_theta > 0:
                delta_theta -= 2 * math.pi
        else:
            raise ValueError(f"Unknown turn_direction: {turn_direction!r}")

        # 3) 在 [theta_start, theta_start + delta_theta] 上等距插值
        thetas = np.linspace(theta_start, theta_start + delta_theta, num=num_points)

        # 4) 生成并可视化
        arc_pts = []
        for t in thetas:
            p = Point(
                center.x + R * math.cos(t),
                center.y + R * math.sin(t)
            )
            arc_pts.append(p)
            # # 可视化测试（如果需要实时展示）
            # self._visualize_arc(custom_map, guide_path, optimized_path, arc_pts, '测试圆弧点')

        return arc_pts

    def _interpolate_line(self, start: Point, end: Point, num_points: int = 8):
        """
        生成从 start 到 end 的线性插值点
        """
        points = []
        for i in range(1, num_points + 1):
            t = i / num_points
            x = start.x + (end.x - start.x) * t
            y = start.y + (end.y - start.y) * t
            points.append(Point(x, y))
        return points

    @staticmethod
    def visualize_comparison(raw_path, optimized_path, smooth_path, game_map):
        """
        三路径对比可视化（Raw → Optimized → Smooth）
        :param raw_path:       原始路径（Point列表，允许为None或[]）
        :param optimized_path: 引导/优化路径（Point列表，允许为None或[]）
        :param smooth_path:    平滑路径（Point列表，允许为None或[]）
        :param game_map:       二维障碍物矩阵（np.ndarray）
        """
        # ———— 画布 & 字体 ————
        fig, ax = plt.subplots(figsize=(12, 10))
        # 如果需要中文标题可以启用下面一行
        plt.rcParams['font.sans-serif'] = 'SimHei'

        # ———— 障碍物底图 ————
        ax.imshow(
            game_map,
            cmap="binary",
            origin="lower",
            extent=[0, game_map.shape[1], 0, game_map.shape[0]]
        )

        # ———— 原始路径：红色虚线 + 关键点 ————
        if raw_path:
            xs = [p.x for p in raw_path]
            ys = [p.y for p in raw_path]
            ax.plot(xs, ys, 'r--', linewidth=2, alpha=0.7, label=f"原始路径 (节点数：{len(raw_path)})")
            ax.scatter(xs, ys, c='red', s=20, marker='o', edgecolors='k', alpha=0.7)

        # ———— 引导/优化路径：绿色点划线 + 标记 ————
        if optimized_path:
            xs = [p.x for p in optimized_path]
            ys = [p.y for p in optimized_path]
            ax.plot(xs, ys, 'g-.', linewidth=2, label=f"引导路径 (节点数：{len(optimized_path)})")
            ax.scatter(xs, ys, c='green', s=20, marker='x')

        # ———— 平滑路径：蓝色实线 + 转折点 ————
        if smooth_path:
            xs = [p.x for p in smooth_path]
            ys = [p.y for p in smooth_path]
            line = ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.9, label=f"优化路径 (节点数：{len(smooth_path)})")
            # 标记中间转折点（跳过起点/终点）
            if len(xs) > 2:
                ax.scatter(
                    xs[1:-1], ys[1:-1],
                    c='#1E90FF',
                    s=20,
                    marker='o',
                    edgecolors='k',
                    zorder=3,
                    label="平滑路径转向点"
                )

        # ———— 起点 & 终点 ————
        # 找到第一个非空路径作为起点来源，最后一个非空路径的终点
        first_path = raw_path or optimized_path or smooth_path
        last_path = smooth_path or optimized_path or raw_path
        if first_path:
            sx, sy = first_path[0].x, first_path[0].y
            ax.scatter(sx, sy, c='lime', s=200, marker='P', edgecolors='k', label="起点")
        if last_path:
            gx, gy = last_path[-1].x, last_path[-1].y
            ax.scatter(gx, gy, c='gold', s=200, marker='*', edgecolors='k', label="终点")

        # ———— 轴 & 网格 & 图例 ————
        ax.set_xlim(0, game_map.shape[1])
        ax.set_ylim(0, game_map.shape[0])
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', color='gray', alpha=0.4)
        ax.set_xlabel("X 坐标", fontsize=12)
        ax.set_ylabel("Y 坐标", fontsize=12)
        ax.set_title("路径规划效果对比: 原始路径 → 引导路径 → 优化路径)", fontsize=14, pad=15)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.show()


