# map_generator.py
import numpy as np
import random
from config import *


def generate_sparse_map(seed=None):
    """生成受种子控制的稀疏连通地图"""
    rng = random.Random(seed)
    game_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)
    protected_coords = create_protected_path(rng)
    add_linear_obstacles(game_map, protected_coords, rng)
    return game_map


def create_protected_path(rng):
    """生成受随机数生成器控制的L型主通道"""
    height, width = MAP_HEIGHT, MAP_WIDTH
    protected = set()

    mid_x = width // 2
    for y in range(height):
        protected.add((y, mid_x))

    link_y = rng.randint(0, height - 1)
    for x in range(mid_x, width):
        protected.add((link_y, x))
    return protected


def add_linear_obstacles(game_map, protected, rng):
    """添加带长度限制和碰撞检测的线性障碍"""
    height, width = game_map.shape
    max_obstacles = int(height * width * MAX_BLACK_RATIO)
    placed = 0

    available = [(y, x) for y in range(height) for x in range(width)
                 if (y, x) not in protected and game_map[y, x] == 0]
    rng.shuffle(available)

    while placed < max_obstacles and available:
        y, x = available.pop()
        valid_directions = []

        # 计算两个方向的可用长度
        h_max = calc_max_length(game_map, protected, (y, x), 'h')
        v_max = calc_max_length(game_map, protected, (y, x), 'v')

        # 筛选符合最小长度要求的方向
        if h_max >= MIN_OBSTACLE_LEN:
            valid_directions.append('h')
        if v_max >= MIN_OBSTACLE_LEN:
            valid_directions.append('v')

        if not valid_directions:
            continue

        # 随机选择方向并计算实际长度
        direction = rng.choice(valid_directions)
        max_len = calc_max_length(game_map, protected, (y, x), direction)
        remaining_space = max_obstacles - placed
        length = min(max_len, remaining_space, MAX_OBSTACLE_LEN)  # 新增MAX限制

        if length < MIN_OBSTACLE_LEN:
            continue

        # 绘制障碍线段
        for i in range(length):
            if direction == 'h':
                dx, dy = x + i, y
            else:
                dx, dy = x, y + i

            # 新增碰撞检测条件
            if dy >= height or dx >= width or (dy, dx) in protected or game_map[dy, dx] == 1:
                break

            game_map[dy, dx] = 1
            placed += 1


def calc_max_length(game_map, protected, start, direction):
    """计算线段最大可延伸长度"""
    y, x = start
    max_len = 0

    if direction == 'h':
        while x + max_len < MAP_WIDTH:
            if (y, x + max_len) in protected or game_map[y, x + max_len] == 1:
                break
            max_len += 1
    else:
        while y + max_len < MAP_HEIGHT:
            if (y + max_len, x) in protected or game_map[y + max_len, x] == 1:
                break
            max_len += 1
    return max_len


