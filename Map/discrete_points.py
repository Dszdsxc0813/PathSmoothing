# discrete_points.py
import numpy as np
import random
from shapely.geometry import Point
from shapely.prepared import prep
from config import MAP_WIDTH, MAP_HEIGHT
from config import MIN_CONNECT_DIS


def generate_discrete_points(game_map, seed=None):
    """生成受种子控制的离散点集合"""
    rng = random.Random(seed)  # 创建独立随机生成器
    W, H = MAP_WIDTH, MAP_HEIGHT
    D = MIN_CONNECT_DIS
    n = int((4 * W * H) / (D ** 2))

    # 预处理可行区域（使用rng进行shuffle）
    free_space = [
        (x, y)
        for y in range(H)
        for x in range(W)
        if game_map[y][x] == 0
    ]
    rng.shuffle(free_space)  # 使用rng替代random

    points = []
    spatial_grid = {}

    # 生成候选点（随机性完全由rng控制）
    for x, y in free_space:
        if len(points) >= n:
            break

        candidate = Point(x, y)

        # 空间索引查询（确定性计算）
        grid_x = x // D
        grid_y = y // D
        conflict = False
        # ...（中间的空间检查逻辑保持不变，无随机性）...

        if not conflict:
            # 精确距离验证（确定性计算）
            valid = all(candidate.distance(p) >= D for p in points)
            if valid:
                points.append(candidate)
                # 更新空间索引（确定性操作）
                key = (grid_x, grid_y)
                if key not in spatial_grid:
                    spatial_grid[key] = []
                spatial_grid[key].append(candidate)

    return points[:n]

# 新增可视化函数：凸显随机的起终点
def visualize_with_ends(game_map, points, start, goal):
    """可视化带起点终点的离散点"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    # 绘制地图背景
    plt.imshow(game_map, cmap='gray_r', origin='lower')

    # 绘制所有离散点
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    plt.scatter(xs, ys, s=20, c='blue', edgecolors='black', label='Discrete Points')

    # 高亮起点终点
    plt.scatter(start.x, start.y, s=150, c='lime', marker='*', edgecolors='black', linewidths=1, label='Start Point')
    plt.scatter(goal.x, goal.y, s=150, c='red', marker='X', edgecolors='black', linewidths=1, label='Goal Point')

    plt.title("Discrete Points with Start/Goal")
    plt.legend(loc='upper right')
    plt.show()
