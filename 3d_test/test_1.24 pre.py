import open3d as o3d
import numpy as np
import copy

# 修改后的可视化函数（不移除法线显示，但不自动估计法线）
def custom_draw_geometry(pcd_list):
    """基础可视化函数"""
    o3d.visualization.draw_geometries(
        pcd_list,
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
        point_show_normal=True  # 保留法线显示（如果存在）
    )

# 简化版数据加载函数
def load_point_cloud(path):
    """基础点云加载"""
    return o3d.io.read_point_cloud(path)

# 简化版预处理
def preprocess_point_cloud(pcd, voxel_size):
    """仅保留降采样和去噪"""
    down_pcd = pcd.voxel_down_sample(voxel_size)
    cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    return down_pcd.select_by_index(ind)

# 加载数据（假设点云文件自带法线）
print("正在加载点云...")
object_raw = load_point_cloud("/home/ncpt-am/local_model/resources/3d/cropped.ply")
scene_raw = load_point_cloud("/home/ncpt-am/local_model/resources/3d/scene.ply")

# 初始位置调整
translation = scene_raw.get_center() - object_raw.get_center()
object_raw.translate(translation * 0.8)

# 颜色标记
object_colored = copy.deepcopy(object_raw).paint_uniform_color([1, 0, 0])
scene_colored = copy.deepcopy(scene_raw).paint_uniform_color([0, 0, 1])

print("\n初始位置可视化：")
custom_draw_geometry([object_colored, scene_colored])

# 预处理执行
voxel_size = 0.05
print(f"\n[预处理] 降采样与去噪 (体素尺寸: {voxel_size})...")
object_down = preprocess_point_cloud(object_raw, voxel_size)
scene_down = preprocess_point_cloud(scene_raw, voxel_size)

# 预处理结果可视化
object_vis = copy.deepcopy(object_down).paint_uniform_color([1, 0.7, 0])
scene_vis = copy.deepcopy(scene_down).paint_uniform_color([0, 0.7, 1])
print("\n预处理结果：")
custom_draw_geometry([object_vis, scene_vis])

# 注意：实际使用时需确保点云包含法线信息，否则后续配准步骤会报错