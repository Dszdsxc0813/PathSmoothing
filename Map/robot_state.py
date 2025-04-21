import math
from dataclasses import dataclass, field
from typing import List, ClassVar
import numpy as np

@dataclass
class Point:
    x: float
    y: float

@dataclass
class RobotState:
    current_pos: Point            # 当前重心坐标
    current_segment_idx: int      # 当前所在路径段索引
    heading_angle: float          # 朝向（相对于 x 轴正向的弧度）
    inflate_distance: float       # 凸壳膨胀距离

    # 原始凸壳顶点（本地坐标系）
    local_convex_hull: ClassVar[List[Point]] = [
        Point(1, -0.5),
        Point(0, -0.5),
        Point(0, 0.5),
        Point(0.5, 2),
        Point(1, 0.5)

    ]

    # 膨胀后的凸壳（实例属性，在 __post_init__ 中自动计算）
    inflate_hull: List[Point] = field(init=False)

    def __post_init__(self):
        # 在对象创建时自动生成膨胀模型
        self.inflate_hull = self.inflate_convex_hull(
            self.local_convex_hull,
            self.inflate_distance
        )

    def compute_centroid(self, points: List[Point]) -> Point:
        """多边形顶点的几何重心 (arithmetic mean)."""
        n = len(points)
        cx = sum(p.x for p in points) / n
        cy = sum(p.y for p in points) / n
        return Point(cx, cy)



    def get_convex_hull(self) -> List[Point]:
        """
        动态计算世界坐标系下的凸壳顶点：
        1) 将 local_convex_hull 的点先平移到以原始重心为 (0,0)；
        2) 再按 heading_angle 旋转；
        3) 最后平移到 current_pos。
        """
        # 1) 计算原始多边形在本地坐标系下的重心
        local_centroid = self.compute_centroid(self.local_convex_hull)

        cos_a = math.cos(self.heading_angle)
        sin_a = math.sin(self.heading_angle)

        world_hull: List[Point] = []
        for p in self.local_convex_hull:
            # 2) 平移到以 local_centroid 为原点的局部坐标
            rx = p.x - local_centroid.x
            ry = p.y - local_centroid.y

            # 3) 按机器人朝向旋转
            x_rot = rx * cos_a - ry * sin_a
            y_rot = rx * sin_a + ry * cos_a

            # 4) 平移到全局坐标 current_pos
            world_x = self.current_pos.x + x_rot
            world_y = self.current_pos.y + y_rot
            world_hull.append(Point(world_x, world_y))

        return world_hull

    def inflate_convex_hull(
            self,
            hull: List[Point],
            buffer_dist: float,
            circle_fix: bool = False,
            circle_samples: int = 12
    ) -> List[Point]:
        """
        用“角平分线”算法对凸多边形 hull 做外膨胀：
          offset = −b_unit * (d / sin(θ/2)),
        其中 b_unit 是内角平分线单位向量，θ 是顶点的内角。
        hull 必须是 CCW（逆时针）顶点顺序。
        """
        inflated = []
        n = len(hull)
        for i in range(n):
            P = hull[i]
            P_prev = hull[i - 1]
            P_next = hull[(i + 1) % n]

            # 边向量 v1=P->P_prev, v2=P->P_next
            v1 = np.array([P_prev.x - P.x, P_prev.y - P.y], dtype=float)
            v2 = np.array([P_next.x - P.x, P_next.y - P.y], dtype=float)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                # 避免退化，保留原点
                inflated.append(Point(P.x, P.y))
                continue

            u1 = v1 / n1
            u2 = v2 / n2

            # 内角 θ
            cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
            theta = math.acos(cos_theta)

            # 角平分线方向（指向内部），b = u1 + u2
            bis = u1 + u2
            bis_len = np.linalg.norm(bis)
            if bis_len < 1e-6:
                # 若顶点几乎平直，用外法线简单膨胀
                # CCW 下，外法线 = u1 旋转 -90°
                normal = np.array([ u1[1], -u1[0] ])
                offset = normal * buffer_dist
            else:
                b_unit = bis / bis_len       # 指向内部
                # 膨胀距离在角平分线上投影：r = d / sin(θ/2)
                r = buffer_dist / math.sin(theta / 2)
                offset = -b_unit * r         # 反方向 => 指向外部

            Qx = P.x + offset[0]
            Qy = P.y + offset[1]
            inflated.append(Point(Qx, Qy))

        return inflated

    def compute_initial_heading_to_align_apex(self, start: Point, end: Point) -> float:
        """根据 start→end 的方向，使尖角相对于重心的连线对齐该方向"""
        # 目标方向向量
        dir_target = np.array([end.x - start.x, end.y - start.y], dtype=float)
        dir_target /= np.linalg.norm(dir_target)

        # 局部重心 → 局部尖角向量
        local_centroid = self.compute_centroid(self.local_convex_hull)
        apex_local = self.local_convex_hull[3]
        dir_apex_local = np.array([apex_local.x - local_centroid.x, apex_local.y - local_centroid.y])
        dir_apex_local /= np.linalg.norm(dir_apex_local)

        # 目标方向相对于局部尖角方向的旋转角度
        dot = np.clip(np.dot(dir_apex_local, dir_target), -1.0, 1.0)
        det = dir_apex_local[0] * dir_target[1] - dir_apex_local[1] * dir_target[0]
        angle = math.atan2(det, dot)
        return angle