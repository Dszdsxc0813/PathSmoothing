# main.py
from Map.discrete_graph import build_discrete_graph
from map_generator import *
from discrete_points import *
from utils import *
from path_planner import *
from path_smoother import PathSmoother
from config import  *

if __name__ == "__main__":
    # 生成50x30的地图（约1500单元格）
    custom_map = generate_sparse_map(seed=6)

    # 生成离散点
    points = generate_discrete_points(custom_map,seed=13)
    print(f"成功生成 {len(points)} 个离散点")
    print("生成点坐标:")
    for index, p in enumerate(points):
        if 20 < p.x < 30:
            print(f"索引: {index}, 坐标: ({p.x}, {p.y})")

    # 构建离散图
    graph = build_discrete_graph(custom_map, points)
    # 示例输出
    sample_point = next(iter(graph))
    print(f"点 {sample_point.wkt} 的连接：")
    for neighbor, dist in graph[sample_point]:
        print(f"  -> {neighbor.wkt} (距离: {dist:.2f})")

    # 随机选择起点终点
    start = points[0]
    goal = points[323]

    # 可视化完整地图
    visualize_discrete_points(custom_map, points)
    visualize_graph(custom_map, graph, points, start, goal)

    # 路径规划与优化
    planner = PathPlanner(graph, custom_map)
    raw_path = planner.dijkstra_path(start, goal)

    # # 初始化优化器（turn_radius与离散点间距一致）
    # smoother = PathSmoother(game_map=custom_map, turn_radius=MIN_OBSTACLE_LEN)
    # # 执行优化
    # smooth_path = smoother.smooth_path(raw_path)

    # 可视化
    if raw_path :
        planner.visualize_paths(raw_path, custom_map)
    else:
        print("路径规划失败！")

    # # 可视化对比
    # PathSmoother.visualize_comparison(raw_path, smooth_path, custom_map)