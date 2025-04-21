# main.py
from src.path_smoothing.map.discrete_graph import build_discrete_graph
from src.path_smoothing.map.map_generator import *
from src.path_smoothing.map.discrete_points import *
from src.path_smoothing.utils.utils import *
from src.path_smoothing.planner.path_planner import *
from src.path_smoothing.smoother.path_smoother import PathSmoother
from src.path_smoothing.config import  *

if __name__ == "__main__":
    # 生成50x30的地图（约1500单元格）
    custom_map = generate_sparse_map(seed=6)

    # 生成离散点
    points = generate_discrete_points(custom_map,seed=12)
    print(f"成功生成 {len(points)} 个离散点")
    print("生成点坐标:")
    for index, p in enumerate(points):
        if 50 < p.y:
            print(f"索引: {index}, 坐标: ({p.x}, {p.y})")
        if 80 < p.x  and p.x >70:
            print(f"索引: {index}, 坐标: ({p.x}, {p.y})")

    # 构建离散图
    graph = build_discrete_graph(custom_map, points)
    # 示例输出
    sample_point = next(iter(graph))
    print(f"点 {sample_point.wkt} 的连接：")
    for neighbor, dist in graph[sample_point]:
        print(f"  -> {neighbor.wkt} (距离: {dist:.2f})")

    # 随机选择起点终点
    start = points[34]
    goal = points[126]

    # 可视化完整离散点及其之间的连接
    visualize_discrete_structure(custom_map, points, None, None, None)
    visualize_discrete_structure(custom_map, points, graph, start, goal)

    # 路径规划与优化
    planner = PathPlanner(graph, custom_map)
    raw_path = planner.dijkstra_path(start, goal)
    print(f"初始路径点如下：")
    for index, i in enumerate(raw_path):
        print(f"索引: {index}, 坐标: ({i.x}, {i.y})")
    guide_path = planner.optimize_path(raw_path)
    # 可视化
    if raw_path:
        visualize_paths(raw_path, guide_path, custom_map)
    else:
        print("路径规划失败！")

    # 路径处理
    # 初始化优化器（turn_radius与离散点间距一致）
    smoother = PathSmoother(game_map=custom_map, turn_radius=MIN_TURN_R)
    # 执行优化
    smooth_path = smoother.smooth_path(raw_path, guide_path, visualize_step=True)

    # 可视化对比
    PathSmoother.visualize_comparison(raw_path, guide_path, smooth_path, custom_map)