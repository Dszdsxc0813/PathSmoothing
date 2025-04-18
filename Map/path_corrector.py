import math
import numpy as np
from matplotlib import pyplot as plt
from robot_state import *

class PathCorrector:
    def __init__(self, turn_radius):
        self.turn_radius = turn_radius  # R

    def _apply_correction(self, robot, segment_start, segment_end,
                          segment_next_end,
                          custom_map,
                          guide_path,
                          optimized_path,
                          segment_after_next_end=None,
                          visualize_steps=None):
        """
        完整路径矫正流程：
        1) 计算圆 O2' 圆心 -> H
        2) 沿 O2' 从 Z 到 H 的圆弧
        3) 计算圆 O3 圆心 -> X
        4) 沿 O3 从 H 到 X 的圆弧
        5) 生成直线 X -> N
        6) 若提供 T (segment_after_next_end)，计算 O4 并沿其圆弧 N -> Z2
        返回整个矫正路径点列表
        """
        # 位置与半径
        if visualize_steps is None:
            visualize_steps = ['centers', 'arc1', 'arc2', 'line', 'arc3', 'full']
        Z = robot.current_pos
        R = self.turn_radius
        alpha = robot.heading_angle

        # 1) 计算圆 O2' 的圆心
        # temp:这需要看角度！！！
        # 计算 AB 方向角 θ
        dx_ab = np.float64(segment_end.x - segment_start.x)
        dy_ab = np.float64(segment_end.y - segment_start.y)
        theta = math.atan2(dy_ab, dx_ab)

        # 计算符号 s = sign(sin(θ - α))
        delta = math.sin(theta - alpha)
        s = 1 if delta >= 0 else -1

        # 修改为显式浮点运算（示例）
        x_O2 = np.float64(Z.x) - np.float64(s * R * np.sin(alpha))
        y_O2 = np.float64(Z.y) + np.float64(s * R * np.cos(alpha))
        O2_center = Point(x_O2, y_O2)

        # 2) 计算圆 O3 的圆心和切入点 X
        O3_center, X = self._compute_O3_and_X(segment_start, segment_end, O2_center)


        # 3) 计算两圆 O2', O3 的切点 H
        H = self.compute_circle_tangent_point(O2_center, O3_center, R)

        # 分步可视化：圆心
        if visualize_steps and 'centers' in visualize_steps:
            self._visualize_centers(custom_map, guide_path, optimized_path,
                                    [O2_center, O3_center, X,H],
                                    ['O2', 'O3', 'X', 'H'])

        # 4) 圆弧 Z->H
        arc1 = self.generate_arc(O2_center, R, Z, H)
        if visualize_steps and 'arc1' in visualize_steps:
            self._visualize_arc(custom_map, guide_path, optimized_path, arc1,'Arc: Z->H')

        # 5) 圆弧 H->X
        arc2 = self.generate_arc(O3_center, R, H, X)
        # 测试完整弧线
        if visualize_steps and 'arc2' in visualize_steps:
            self._visualize_arc(custom_map, guide_path, optimized_path, arc2, 'Arc: H->X')

        # 6) 直线 X->N
        N = self.compute_blend_start(segment_start, segment_end, segment_next_end, R)
        line1 = self.generate_line(X, N)
        if visualize_steps and 'line' in visualize_steps:
            self._visualize_line(custom_map, guide_path, optimized_path, line1, 'Line: X->N')

        path = []
        path.extend(arc1)
        path.extend(arc2)
        path.extend(line1)

        # --- 7) 生成圆弧 N->Z2 ---
        if segment_after_next_end:
            O4_center = self.compute_blend_center(segment_end,
                                                  segment_next_end,
                                                  segment_after_next_end, R)
            Z2 = self.compute_blend_start(segment_end,
                                          segment_next_end,
                                          segment_after_next_end, R)
            arc3 = self.generate_arc(O4_center, R, N, Z2)
            if visualize_steps and 'arc3' in visualize_steps:
                self._visualize_arc(custom_map, guide_path, optimized_path, arc3, 'Circle O4 Arc: N->Z2')
            path.extend(arc3)

            # 全量可视化
        # if visualize_steps and 'full' in visualize_steps:
        #     self.visualize_correction(robot,
        #                             segment_start,
        #                             segment_end,
        #                             segment_next_end,
        #                             segment_after_next_end,
        #                             custom_map)

        return path

    def _compute_O3_and_X(self, A, B, O2_center):
        """
        根据论文公式 10-16，计算圆 O3 圆心和切入点 X
        且保证 O2_center 和 O3_center 分布在路径段 AB 的两侧
        """
        R = self.turn_radius
        x1 = np.float64(A.x)
        y1 = np.float64(A.y)
        x2 = np.float64(B.x)
        y2 = np.float64(B.y)
        dx, dy = x2 - x1, y2 - y1

        # 1) 计算斜率 k
        k = dy / dx if abs(dx) > 1e-6 else float('inf')

        # 2) AB 的标准式：a_ab * x + b_ab * y + c_ab = 0
        a_ab = -dy
        b_ab = dx
        c_ab = dy * x1 - dx * y1

        # 3) 计算 O2 到 AB 的符号距离（不除以 sqrt(a^2 + b^2)，保留符号即可）
        x_O2, y_O2 = O2_center.x, O2_center.y
        sign_O2 = 1 if (a_ab * x_O2 + b_ab * y_O2 + c_ab) >= 0 else -1

        # 4) 平移偏移量（方向依据 sign_O2）
        if k != float('inf'):
            offset_x = -sign_O2 * R * k / math.sqrt(1 + k**2)
            offset_y =  sign_O2 * R       / math.sqrt(1 + k**2)
        else:
            # 垂直线特殊处理：水平平移
            offset_x = sign_O2 * R
            offset_y = 0

        # 5) 构造平移直线 L：y = k*(x - (x1 + offset_x)) + y1 + offset_y
        # 即：y = k*x + (y1 - k*(x1 + offset_x) + offset_y)
        if k != float('inf'):
            L_k = k
            L_b = y1 - k * (x1 + offset_x) + offset_y
        else:
            # 垂直线：x = x1 + offset_x
            L_k = None
            L_x_const = x1 + offset_x

        # 6) 求 L 与圆 O2 的交点（代入直线到圆方程中）
        # 圆心 O2，半径 2R（因为 O2 和 O3 间距为 2R）
        r = 2 * R
        if k != float('inf'):
            A_quad = 1 + k**2
            B_quad = 2 * (k * (L_b - y_O2) - x_O2)
            C_quad = x_O2**2 + (L_b - y_O2)**2 - r**2

            disc = B_quad**2 - 4 * A_quad * C_quad
            if disc < 0:
                raise ValueError("No intersection between O2' and L")
            sqrt_disc = math.sqrt(disc)
            x_a = (-B_quad + sqrt_disc) / (2 * A_quad)
            x_b = (-B_quad - sqrt_disc) / (2 * A_quad)
            y_a = L_k * x_a + L_b
            y_b = L_k * x_b + L_b
        else:
            # 垂直线：x = L_x_const，代入圆方程 (x - x_O2)^2 + (y - y_O2)^2 = r^2
            x_a = x_b = L_x_const
            delta = r**2 - (x_a - x_O2)**2
            if delta < 0:
                raise ValueError("No intersection between O2' and vertical L")
            sqrt_delta = math.sqrt(delta)
            y_a = y_O2 + sqrt_delta
            y_b = y_O2 - sqrt_delta

        # 7) 从候选中选出与 O2 在 AB 的异侧点
        candidates = [(x_a, y_a), (x_b, y_b)]
        x_O3 = y_O3 = None
        for xc, yc in candidates:
            sign_c = a_ab * xc + b_ab * yc + c_ab
            if sign_O2 * sign_c < 0:
                x_O3, y_O3 = xc, yc
                break
        if x_O3 is None:
            x_O3, y_O3 = candidates[1]  # 兜底

        O3_center = Point(x_O3, y_O3)

        # 8) 切入路径AB的圆O3上的切入点 X
        # 交点 X：圆 O3 与 AB 直线
        A_x = 1 + (k**2 if k != float('inf') else 0)
        B_x = -2 * x_O3 - 2 * k ** 2 * x1 + 2 * k * y1 - 2 * k * y_O3
        C_x = x_O3**2 + k**2 * x1**2 - 2 * k * x1 * y1 + 2 * k * y_O3 * x1 + y1 ** 2 - 2 * y_O3 * y1 + y_O3 ** 2 - R**2
        disc2 = B_x**2 - 4 * A_x * C_x
        if disc2 < 0:
            raise ValueError("No intersection for O3 and AB")
        sqrt2 = math.sqrt(disc2)
        sols = [(-B_x + sqrt2) / (2 * A_x), (-B_x - sqrt2) / (2 * A_x)]
        X_candidates = []
        for xX in sols:
            yX = k * (xX - x1) + y1
            # 只保留 0<=t<=1 的点
            t = (xX - x1)/dx if abs(dx) > abs(dy) else (yX-y1)/dy
            if 0 <= t <= 1:
                X_candidates.append(Point(xX, yX))
        if not X_candidates:
            raise ValueError("No valid X on AB segment")
        return O3_center, X_candidates[0]


    def compute_circle_tangent_point(self, O2_center, O3_center, R):
        """
        两圆外切点 H
        """
        x1 = np.float64(O2_center.x)
        y1 = np.float64(O2_center.y)
        x2 = np.float64(O3_center.x)
        y2 = np.float64(O3_center.y)

        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        return Point(x1 + R * ux, y1 + R * uy)

    def compute_blend_start(self, A: Point, B: Point, C: Point, R: float) -> Point:
        """
        计算预期转向点 N
        """
        # 确保所有计算使用 numpy.float64
        def to_vec(P, Q):
            return np.float64(Q.x - P.x), np.float64(Q.y - P.y)
        def norm(v):
            return np.hypot(np.float64(v[0]), np.float64(v[1]))
        def scale(v, s):
            return np.float64(v[0] * s), np.float64(v[1] * s)
        def add(u, v):
            return np.float64(u[0] + v[0]), np.float64(u[1] + v[1])
        def dot(u, v):
            return np.float64(u[0] * v[0] + u[1] * v[1])
        def cross(u, v):
            return np.float64(u[0] * v[1] - u[1] * v[0])

        u1 = to_vec(A,B); L1=norm(u1); u1=(u1[0]/L1,u1[1]/L1)
        u2 = to_vec(B,C); L2=norm(u2); u2=(u2[0]/L2,u2[1]/L2)
        n1 = (-u1[1],u1[0]);
        if dot(n1,to_vec(B,C))<0: n1=(-n1[0],-n1[1])
        n2 = (-u2[1],u2[0]);
        if dot(n2,to_vec(B,A))<0: n2=(-n2[0],-n2[1])
        P1 = add((A.x,A.y), scale(n1,R))
        P2 = add((B.x,B.y), scale(n2,R))
        denom = cross(u1,u2)
        if abs(denom)<1e-9: raise ValueError("Parallel")
        t1=cross((P2[0]-P1[0],P2[1]-P1[1]),u2)/denom
        O_blend=(P1[0]+t1*u1[0],P1[1]+t1*u1[1])
        w=(A.x-O_blend[0],A.y-O_blend[1])
        a=dot(u1,u1); b=2*dot(u1,w); c=dot(w,w)-R*R
        disc=b*b-4*a*c
        if disc<0: raise ValueError("No N")
        sd=math.sqrt(disc)
        s1=(-b+sd)/(2*a); s2=(-b-sd)/(2*a)
        cands=[]
        for s in (s1,s2):
            if 0<=s<=L1:
                cands.append((A.x+s*u1[0],A.y+s*u1[1]))
        if not cands: raise ValueError("No N on AB")
        def dist2(p,Q): return (p[0]-Q.x)**2+(p[1]-Q.y)**2
        N=max(cands,key=lambda p:dist2(p,B))
        return Point(*N)

    def compute_blend_center(self, A: Point, B: Point, C: Point, R: float) -> Point:
        """
        计算平滑转弯圆心 O4
        """
        # 复用 compute_blend_start 求 O_blend
        # 直接 compute offset lines intersection
        # 确保所有计算使用 numpy.float64
        def to_vec(P, Q):
            return np.float64(Q.x - P.x), np.float64(Q.y - P.y)
        def norm(v):
            return np.hypot(np.float64(v[0]), np.float64(v[1]))
        def scale(v, s):
            return np.float64(v[0] * s), np.float64(v[1] * s)
        def add(u, v):
            return np.float64(u[0] + v[0]), np.float64(u[1] + v[1])
        def dot(u, v):
            return np.float64(u[0] * v[0] + u[1] * v[1])
        def cross(u, v):
            return np.float64(u[0] * v[1] - u[1] * v[0])

        u1=to_vec(A,B);L1=norm(u1);u1=(u1[0]/L1,u1[1]/L1)
        u2=to_vec(B,C);L2=norm(u2);u2=(u2[0]/L2,u2[1]/L2)
        n1=(-u1[1],u1[0]);
        if dot(n1,to_vec(B,C))<0:n1=(-n1[0],-n1[1])
        n2=(-u2[1],u2[0]);
        if dot(n2,to_vec(B,A))<0:n2=(-n2[0],-n2[1])
        P1=add((A.x,A.y),scale(n1,R))
        P2=add((B.x,B.y),scale(n2,R))
        denom=cross(u1,u2)
        if abs(denom)<1e-9:raise ValueError("Parallel")
        t1=cross((P2[0]-P1[0],P2[1]-P1[1]),u2)/denom
        O_blend=(P1[0]+t1*u1[0],P1[1]+t1*u1[1])
        return Point(*O_blend)

    def generate_arc(self, center: Point, R: float,
                     start_pt: Point, end_pt: Point,
                     num_points: int = 16):
        """
        生成圆弧点
        """
        theta_start = np.arctan2(np.float64(start_pt.y - center.y),
                                 np.float64(start_pt.x - center.x))
        theta_end = np.arctan2(np.float64(end_pt.y - center.y),
                               np.float64(end_pt.x - center.x))

        if theta_end < theta_start:
            theta_end += 2 * math.pi

        thetas = np.linspace(theta_start, theta_end, num=num_points)
        # math.cos可以替换成np.cos
        return [Point(center.x + R * math.cos(t),
                      center.y + R * math.sin(t))
                for t in thetas]

    def generate_line(self, p1: Point, p2: Point,
                      num_points: int = 16):
        """
        线性插值生成
        """
        # t可以替换成np.float64(t)
        return [Point(p1.x + (p2.x - p1.x) * t,
                      p1.y + (p2.y - p1.y) * t)
                for t in np.linspace(0, 1, num_points)]

    def _visualize_centers(self, game_map, guide_path, optimized_path, centers, labels):
        # ========================== 画布初始化 ==========================
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体支持

        # ====================== 障碍物地图可视化 ========================
        ax.imshow(game_map, cmap="binary",
                  origin="lower",
                  extent=[0, game_map.shape[1], 0, game_map.shape[0]])

        # ====================== 路径数据预处理 ========================
        def get_path_coords(path):
            return ([p.x for p in path], [p.y for p in path]) if path else (None, None)

        # 提取各路径坐标
        guide_x, guide_y = get_path_coords(guide_path)
        opt_x, opt_y = get_path_coords(optimized_path)

        # ====================== 路径可视化层 ========================
        # ---- 引导路径（绿色点划线）----
        if guide_x and guide_y:
            ax.plot(guide_x, guide_y, 'g-.', linewidth=2,
                    label=f"引导路径 (节点数:{len(guide_path)})")
            ax.scatter(guide_x, guide_y, s=20, c='green', marker='x')

        # ---- 优化路径（蓝色实线 + 动态点）----
        if opt_x and opt_y:
            ax.plot(opt_x, opt_y, 'b-', linewidth=2, alpha=0.8,
                    label=f"优化路径 (节点数:{len(optimized_path)})")
            # 动态点标注（最后一个点为机器人当前位置）
            ax.scatter(opt_x, opt_y, s=15, c='cyan', marker='o',
                       edgecolors='k')

        # ==================== 起点终点标记层 ======================
        if guide_path:
            all_paths = [p for p in [guide_path] if p]
            start_point = all_paths[0][0] if all_paths else None
            goal_point = all_paths[-1][-1] if all_paths else None
            if start_point:
                ax.scatter(start_point.x, start_point.y, c='lime', s=200,
                           marker='P', edgecolors='k', label="起点")
            if goal_point:
                ax.scatter(goal_point.x, goal_point.y, c='gold', s=200,
                           marker='*', edgecolors='k', label="终点")
        # ==================== 圆心 ============================
        for cpt, lab in zip(centers, labels):
            ax.scatter(cpt.x, cpt.y, s=30, marker='X', label=lab)
        ax.set_title('圆心位置', fontsize=14)
        # ====================== 坐标轴装饰 ========================
        ax.set_xlim(0, game_map.shape[1])
        ax.set_ylim(0, game_map.shape[0])
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', color='gray', alpha=0.4)
        ax.set_xlabel("X 坐标", fontsize=12)
        ax.set_ylabel("Y 坐标", fontsize=12)
        ax.set_title("路径规划效果对比: 引导路径 → 优化路径", fontsize=14, pad=15)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    def _visualize_arc(self, game_map, guide_path, optimized_path, arc_pts, title):
        # ========================== 画布初始化 ==========================
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体支持

        # ====================== 障碍物地图可视化 ========================
        ax.imshow(game_map, cmap="binary",
                  origin="lower",
                  extent=[0, game_map.shape[1], 0, game_map.shape[0]])

        # ====================== 路径数据预处理 ========================
        def get_path_coords(path):
            return ([p.x for p in path], [p.y for p in path]) if path else (None, None)

        # 提取各路径坐标
        guide_x, guide_y = get_path_coords(guide_path)
        opt_x, opt_y = get_path_coords(optimized_path)

        # ====================== 路径可视化层 ========================
        # ---- 引导路径（绿色点划线）----
        if guide_x and guide_y:
            ax.plot(guide_x, guide_y, 'g-.', linewidth=2,
                    label=f"引导路径 (节点数:{len(guide_path)})")
            ax.scatter(guide_x, guide_y, s=20, c='green', marker='x')

        # ---- 优化路径（蓝色实线 + 动态点）----
        if opt_x and opt_y:
            ax.plot(opt_x, opt_y, 'b-', linewidth=2, alpha=0.8,
                    label=f"优化路径 (节点数:{len(optimized_path)})")
            # 动态点标注（最后一个点为机器人当前位置）
            ax.scatter(opt_x, opt_y, s=15, c='cyan', marker='o',
                       edgecolors='k')

        xs = [p.x for p in arc_pts]
        ys = [p.y for p in arc_pts]
        ax.plot(xs, ys, 'g-', linewidth=2, label=title)
        ax.scatter(xs, ys, s=30, c='cyan', marker='o', edgecolors='k')
        # ==================== 起点终点标记层 ======================
        if guide_path:
            all_paths = [p for p in [guide_path] if p]
            start_point = all_paths[0][0] if all_paths else None
            goal_point = all_paths[-1][-1] if all_paths else None
            if start_point:
                ax.scatter(start_point.x, start_point.y, c='lime', s=200,
                           marker='P', edgecolors='k', label="起点")
            if goal_point:
                ax.scatter(goal_point.x, goal_point.y, c='gold', s=200,
                           marker='*', edgecolors='k', label="终点")

        # ====================== 坐标轴装饰 ========================
        ax.set_xlim(0, game_map.shape[1])
        ax.set_ylim(0, game_map.shape[0])
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', color='gray', alpha=0.4)
        ax.set_xlabel("X 坐标", fontsize=12)
        ax.set_ylabel("Y 坐标", fontsize=12)
        ax.set_title("路径规划效果对比: 引导路径 → 优化路径", fontsize=14, pad=15)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    def _visualize_line(self, game_map, guide_path, optimized_path, line_pts, title):
        # ========================== 画布初始化 ==========================
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体支持

        # ====================== 障碍物地图可视化 ========================
        ax.imshow(game_map, cmap="binary",
                  origin="lower",
                  extent=[0, game_map.shape[1], 0, game_map.shape[0]])

        # ====================== 路径数据预处理 ========================
        def get_path_coords(path):
            return ([p.x for p in path], [p.y for p in path]) if path else (None, None)

        # 提取各路径坐标
        guide_x, guide_y = get_path_coords(guide_path)
        opt_x, opt_y = get_path_coords(optimized_path)

        # ====================== 路径可视化层 ========================
        # ---- 引导路径（绿色点划线）----
        if guide_x and guide_y:
            ax.plot(guide_x, guide_y, 'g-.', linewidth=2,
                    label=f"引导路径 (节点数:{len(guide_path)})")
            ax.scatter(guide_x, guide_y, s=20, c='green', marker='x')

        # ---- 优化路径（蓝色实线 + 动态点）----
        if opt_x and opt_y:
            ax.plot(opt_x, opt_y, 'b-', linewidth=2, alpha=0.8,
                    label=f"优化路径 (节点数:{len(optimized_path)})")
            # 动态点标注（最后一个点为机器人当前位置）
            ax.scatter(opt_x, opt_y, s=15, c='cyan', marker='o',
                       edgecolors='k')

        xs = [p.x for p in line_pts]
        ys = [p.y for p in line_pts]
        ax.plot(xs, ys, 'g-', linewidth=2, label=title)
        ax.scatter(xs, ys, s=30, c='lime', marker='o', edgecolors='k')
        # ==================== 起点终点标记层 ======================
        if guide_path:
            all_paths = [p for p in [guide_path] if p]
            start_point = all_paths[0][0] if all_paths else None
            goal_point = all_paths[-1][-1] if all_paths else None
            if start_point:
                ax.scatter(start_point.x, start_point.y, c='lime', s=200,
                           marker='P', edgecolors='k', label="起点")
            if goal_point:
                ax.scatter(goal_point.x, goal_point.y, c='gold', s=200,
                           marker='*', edgecolors='k', label="终点")

        # ====================== 坐标轴装饰 ========================
        ax.set_xlim(0, game_map.shape[1])
        ax.set_ylim(0, game_map.shape[0])
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', color='gray', alpha=0.4)
        ax.set_xlabel("X 坐标", fontsize=12)
        ax.set_ylabel("Y 坐标", fontsize=12)
        ax.set_title("路径规划效果对比: 引导路径 → 优化路径", fontsize=14, pad=15)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

