# utils.py
import matplotlib.pyplot as plt
import numpy as np
from src.path_smoothing.robot.robot_state import *

import matplotlib.pyplot as plt

def visualize_discrete_structure(game_map, points, graph=None, start=None, goal=None):
    """
    离散点 / 离散图 可视化
    :param game_map: 二维障碍物矩阵（np.ndarray）
    :param points:   Point 列表
    :param graph:    可选 dict，键为 Point，值为 List[(Point, weight)]；如果提供，则绘制连边
    :param start:    可选 Point，起点（仅在 graph 模式下绘制）
    :param goal:     可选 Point，终点（仅在 graph 模式下绘制）
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    # 障碍物底图
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体支持
    ax.imshow(
        game_map,
        cmap="binary",
        origin="lower",
        extent=[0, game_map.shape[1], 0, game_map.shape[0]]
    )

    # 绘制节点
    if points:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        label = f"节点 (共 {len(points)} 个)"
        ax.scatter(
            xs, ys,
            s=30,
            c='red',
            marker='o',
            edgecolors='k',
            label=label
        )

    # 绘制起点/终点（仅当 graph 模式并提供 start/goal）
    if graph and start and goal:
        ax.scatter(
            start.x, start.y,
            s=200, c='lime',
            marker='P', edgecolors='k',
            label="起点"
        )
        ax.scatter(
            goal.x, goal.y,
            s=200, c='gold',
            marker='*', edgecolors='k',
            label="终点"
        )

    # 轴、网格、标签、图例
    ax.set_xlim(0, game_map.shape[1])
    ax.set_ylim(0, game_map.shape[0])
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', color='gray', alpha=0.4)
    ax.set_xlabel("X 坐标", fontsize=12)
    ax.set_ylabel("Y 坐标", fontsize=12)

    title = "离散图可视化" if graph else f"离散点生成验证 (共 {len(points)} 个)"
    ax.set_title(title, fontsize=14, pad=10)

    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()



def visualize_paths(raw_path, guide_path, game_map):
    """
    三路径对比可视化（支持路径缺失场景）
    :param raw_path:       原始路径（Point列表，允许为None）
    :param guide_path: 引导路径（Point列表，允许为None）
    :param game_map:       二维障碍物矩阵（np.ndarray）
    """
    # ========================== 画布初始化 ==========================
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体支持

    # ====================== 障碍物地图可视化 ========================
    # 使用灰度色图：0=白色（可行区域），1=黑色（障碍物）
    ax.imshow(game_map, cmap="binary",
              origin="lower",  # 坐标系原点在左下角
              extent=[0, game_map.shape[1], 0, game_map.shape[0]])

    # ====================== 路径数据预处理 ========================
    def get_path_coords(path):
        """安全提取路径坐标（处理空路径）"""
        return ([p.x for p in path], [p.y for p in path]) if path else (None, None)

    # 提取各路径坐标
    raw_x, raw_y = get_path_coords(raw_path)
    opt_x, opt_y = get_path_coords(guide_path)

    # ====================== 路径可视化层 ========================
    # ---- 原始路径（红色虚线）----
    if raw_x and raw_y:
        ax.plot(raw_x, raw_y, 'r--',
                linewidth=1.5, alpha=0.7,
                label=f"原始路径 (节点数:{len(raw_path)})")
        # 绘制离散点，使用红色的 'x' 标注
        xs = [p.x for p in raw_path]
        ys = [p.y for p in raw_path]
        plt.scatter(xs, ys, s=20, c='red', marker='o')

    # ---- 引导路径（绿色点划线）----
    if opt_x and opt_y:
        ax.plot(opt_x, opt_y, 'g-.',
                linewidth=2, markersize=6,
                label=f"引导路径 (节点数:{len(guide_path)})")
        # 绘制离散点，使用绿色的圆点标注
        xs = [p.x for p in guide_path]
        ys = [p.y for p in guide_path]
        plt.scatter(xs, ys, s=20, c='green', marker='x')

    # ==================== 起点终点标记层 ======================
    if raw_path:
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
    ax.set_aspect('equal')  # 等比例坐标轴
    ax.grid(True, linestyle=':', color='gray', alpha=0.4)
    ax.set_xlabel("X 坐标", fontsize=12)
    ax.set_ylabel("Y 坐标", fontsize=12)
    ax.set_title("路径规划效果对比: 原始路径 -> 引导路径",
                 fontsize=14, pad=15)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()



def visualize_dynamic_paths(guide_path, optimized_path, game_map):
    """
    三路径对比可视化（支持路径缺失场景）
    :param raw_path:       原始路径（Point列表，允许为None）
    :param guide_path:     引导路径（Point列表，允许为None）
    :param optimized_path: 优化路径（Point列表，允许为None）
    :param game_map:       二维障碍物矩阵（np.ndarray）
    """
    # ========================== 画布初始化 ==========================
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体支持

    # ====================== 障碍物地图可视化 ========================
    ax.imshow(
        game_map,
        cmap="binary",
        origin="lower",
        extent=[0, game_map.shape[1], 0, game_map.shape[0]]
    )

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
        ax.plot(opt_x, opt_y, 'b--', linewidth=1.5, alpha=0.8,
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

    def visualize_convex_hulls(convex_hull: list[Point], expanded_hull: list[Point]):
        """
        可视化凸壳模型及其扩张后的结果。

        Args:
            convex_hull: 原始凸壳模型的顶点列表。
            expanded_hull: 扩张后的凸壳模型的顶点列表。
        """
        # 提取原始凸壳的坐标
        original_x = [p.x for p in convex_hull]
        original_y = [p.y for p in convex_hull]
        original_x.append(original_x[0])  # 闭合多边形
        original_y.append(original_y[0])

        # 提取扩张后的凸壳的坐标
        expanded_x = [p.x for p in expanded_hull]
        expanded_y = [p.y for p in expanded_hull]
        expanded_x.append(expanded_x[0])  # 闭合多边形
        expanded_y.append(expanded_y[0])

        # 创建图形
        plt.figure(figsize=(10, 10))
        plt.plot(original_x, original_y, 'b-', label='Original Convex Hull')
        plt.plot(expanded_x, expanded_y, 'r-', label='Expanded Convex Hull')
        plt.scatter([p.x for p in convex_hull], [p.y for p in convex_hull], c='blue', marker='o',
                    label='Original Vertices')
        plt.scatter([p.x for p in expanded_hull], [p.y for p in expanded_hull], c='red', marker='x',
                    label='Expanded Vertices')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Convex Hull Expansion Visualization')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()