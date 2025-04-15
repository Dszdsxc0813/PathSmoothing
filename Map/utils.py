# utils.py
import matplotlib.pyplot as plt
import numpy as np

# 基础地图可视化
def plot_map(map_matrix):
    """可视化地图（1像素=1单元格）"""
    plt.figure(figsize=(12, 8))
    plt.imshow(map_matrix, cmap="binary", interpolation="none")
    plt.xticks([]), plt.yticks([])
    plt.show()

# 离散点生成可视化验证
def visualize_discrete_points(game_map, points):
    plt.figure(figsize=(12, 8))

    # 绘制地图
    plt.imshow(game_map, cmap='gray_r', origin='lower')

    # 绘制离散点
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    plt.scatter(xs, ys, s=10, c='red', marker='o')

    plt.title(f"Discrete Points ({len(points)} nodes)")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# 构建离散图可视化验证
def visualize_graph(game_map, graph, points, start, goal):
    # visualize_with_ends(custom_map, points, start, goal)  # 新增调用
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.imshow(game_map, cmap='gray_r', origin='lower')

    # 绘制所有连接
    for p in graph:
        px, py = p.x, p.y
        for q, _ in graph[p]:
            qx, qy = q.x, q.y
            plt.plot([px, qx], [py, qy], 'b-', alpha=0.3, linewidth=0.5)

    # 绘制节点
    xs = [p.x for p in graph]
    ys = [p.y for p in graph]
    plt.scatter(xs, ys, s=20, c='red', edgecolors='black')

    plt.scatter(start.x, start.y, s=150, c='lime', marker='*', edgecolors='black', linewidths=1, label='Start Point')
    plt.scatter(goal.x, goal.y, s=150, c='red', marker='X', edgecolors='black', linewidths=1, label='Goal Point')

    plt.title("Discrete Graph Visualization")
    plt.legend(loc='upper right')
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
        start_point = next((p for p in [raw_path[0] if raw_path else None, guide_path[0] if guide_path else None] if p), None)
        goal_point = next((p[-1] for p in [raw_path, guide_path] if p), None)
        if start_point:
            ax.scatter(start_point.x, start_point.y,
                       c='lime', s=200, marker='P', edgecolors='k', label="Start")
        if goal_point:
            ax.scatter(goal_point.x, goal_point.y,
                       c='gold', s=200, marker='*', edgecolors='k', label="Goal")

    # ====================== 坐标轴装饰 ========================
    ax.set_xlim(0, game_map.shape[1])
    ax.set_ylim(0, game_map.shape[0])
    ax.set_aspect('equal')  # 等比例坐标轴
    ax.grid(True, linestyle=':', color='gray', alpha=0.4)
    ax.set_xlabel("X 坐标", fontsize=12)
    ax.set_ylabel("Y 坐标", fontsize=12)
    ax.set_title("路径规划效果对比: 原始路径 → 引导路径 → 平滑路径",
                 fontsize=14, pad=15)

    # ====================== 图例与输出 ========================
    plt.tight_layout()
    plt.legend()
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