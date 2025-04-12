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
