# discrete_graph.py
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString
from tqdm import tqdm  # 进度条可视化


def build_discrete_graph(game_map, points, m=4):
    """构建离散连接图

    参数：
    game_map -- 二维障碍物矩阵（0=可行，1=障碍）
    points   -- 离散点集合（Shapely Point对象）
    m        -- 每个点连接的最多邻接点数（论文中取3）

    返回：
    graph_dict -- 邻接表字典 {Point: [(相邻点, 距离), ...]}
    """
    # 将点转换为坐标数组加速计算
    coords = np.array([[p.x, p.y] for p in points])
    kdtree = KDTree(coords)  # 构建KD树加速近邻查询

    graph = {}
    obstacle_lines = precompute_obstacles(game_map)  # 预处理障碍物边界

    for i, p in enumerate(tqdm(points, desc="Building Graph")):
        # 查询最近的m+5个点（扩大候选集）
        distances, indices = kdtree.query(p.coords[0], k=m + 5)

        neighbors = []
        for idx, dist in zip(indices, distances):
            if idx == i: continue  # 排除自身

            q = points[idx]
            if is_valid_connection(p, q, game_map, obstacle_lines):
                neighbors.append((q, dist))

            if len(neighbors) >= m:  # 达到最大连接数
                break

        graph[p] = neighbors

    return graph


def precompute_obstacles(game_map):
    """预处理障碍物边界（提升碰撞检测速度）"""
    from shapely.geometry import Polygon
    obstacles = []
    height, width = game_map.shape

    # 识别连续障碍区域
    for y in range(height):
        for x in range(width):
            if game_map[y][x] == 1:
                # 创建障碍物矩形（每个格子视为1x1的障碍）
                obstacles.append(Polygon([
                    (x - 0.5, y - 0.5), (x + 0.5, y - 0.5),
                    (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)
                ]))

    return obstacles


def is_valid_connection(p1, p2, game_map, obstacles):
    """验证两点连线是否跨越障碍"""
    line = LineString([p1, p2])

    # 快速碰撞检测（Shapely空间谓词）
    for obst in obstacles:
        if line.intersects(obst):
            return False

    # 精确栅格检测（双重验证）
    path = bresenham_line(p1, p2)
    for (x, y) in path:
        if game_map[int(y)][int(x)] == 1:
            return False
    return True


def bresenham_line(p1, p2):
    """Bresenham算法生成两点间直线路径"""
    x1, y1 = round(p1.x), round(p1.y)
    x2, y2 = round(p2.x), round(p2.y)

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = abs(y2 - y1)
    error = dx // 2
    ystep = 1 if y1 < y2 else -1

    path = []
    y = y1
    for x in range(x1, x2 + 1):
        coord = (y, x) if steep else (x, y)
        path.append(coord)
        error -= dy
        if error < 0:
            y += ystep
            error += dx
    return path

