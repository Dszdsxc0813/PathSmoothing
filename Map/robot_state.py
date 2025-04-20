import math
from dataclasses import dataclass, field


@dataclass
class Point:
    x: float
    y: float

@dataclass
class RobotState:
    current_pos: Point            # 当前重心坐标
    current_segment_idx: int      # 当前所在路径段索引
    heading_angle: float          # 朝向（相对于 x 轴正向的弧度）

