import math
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

@dataclass
class RobotState:
    current_pos: Point       # 当前坐标 (x0, y0)
    current_segment_idx: int # 当前所在路径段的索引（路径段为 path[i] -> path[i+1]）
    heading_angle: float     # 当前朝向角α（相对于y轴正方向的弧度）