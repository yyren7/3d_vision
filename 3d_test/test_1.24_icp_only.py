# %% [markdown]
# ## 点云配准流程（直接ICP版）
# 本 Notebook 实现基于多阶段ICP的直接配准流程

# %% [markdown]
# ### 1. 导入依赖库

# %%
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

# %% [markdown]
# ### 2. 定义增强可视化函数

# %%
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

# %% [markdown]
# ### 3. 数据加载与初始化

# %%
# 加载点云（注意修改文件路径）
print("正在加载点云...")
object_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/cropped_1.ply")
scene_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/scene_cropped_1.ply")

# 初始位置调整（重要！需要手动设置合理初始位置）
translation = scene_raw.get_center() - object_raw.get_center()
object_raw.translate(translation * 0.5)  # 调整平移比例为50%

# 颜色标记
object_colored = copy.deepcopy(object_raw).paint_uniform_color([1, 0, 0])  # 红色为物体
scene_colored = copy.deepcopy(scene_raw).paint_uniform_color([0, 0, 1])    # 蓝色为场景

print("\n初始位置可视化：")
custom_draw_geometry([object_colored, scene_colored])

# %% [markdown]
# ### 4. 数据预处理

# %%
# 统一降采样参数
voxel_size = 0.02  # 调整为更细的采样粒度

# 降采样处理
print("\n[预处理] 降采样...")
object_down = object_raw.voxel_down_sample(voxel_size)
scene_down = scene_raw.voxel_down_sample(voxel_size)

# 去噪处理
print("[预处理] 统计离群点去除...")
object_down, _ = object_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
scene_down, _ = scene_down.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.8)

# 法线估计（关键步骤）
print("[预处理] 法线估计...")
object_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
scene_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 统一法线方向（重要！确保法线方向一致性）
object_down.orient_normals_to_align_with_direction([0, 0, 1])
scene_down.orient_normals_to_align_with_direction([0, 0, 1])

# 可视化预处理结果
object_vis = copy.deepcopy(object_down).paint_uniform_color([1, 0.7, 0])
scene_vis = copy.deepcopy(scene_down).paint_uniform_color([0, 0.7, 1])
print("\n预处理结果：")
custom_draw_geometry([object_vis, scene_vis])

# %% [markdown]
# ### 5. 多阶段ICP精配准

# %%
# ICP阶段参数设置（优化参数）
icp_stages = [
    {"max_iter": 100, "max_dist": 100.0}, # 第二阶段：中等精度
    {"max_iter": 200, "max_dist": 60.0}, # 第三阶段：精细配准
    {"max_iter": 400, "max_dist": 50.0}, # 第三阶段：精细配准
    {"max_iter": 800, "max_dist": 40.0}, # 第三阶段：精细配准
    {"max_iter": 1600, "max_dist": 30.0}, # 第三阶段：精细配准
    {"max_iter": 3200, "max_dist": 5.0}, # 第三阶段：精细配准
    {"max_iter": 6400, "max_dist": 2.0}, # 第三阶段：精细配准
]

current_trans = np.identity(4)  # 初始化为单位矩阵

for stage_idx, stage in enumerate(icp_stages):
    print(f"\n[阶段 {stage_idx+1}] ICP配准 | 最大距离: {stage['max_dist']}m | 最大迭代: {stage['max_iter']}")
    
    result = o3d.pipelines.registration.registration_icp(
        object_down, scene_down,
        max_correspondence_distance=stage["max_dist"],
        init=current_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=stage["max_iter"]
        )
    )
    
    current_trans = result.transformation
    print(f"阶段结果: 匹配RMSE = {result.inlier_rmse:.4f} | 适应度 = {result.fitness:.3f}")

# %% [markdown]
# ### 6. 结果验证与可视化

# %%
# 应用最终变换
object_final = copy.deepcopy(object_down).transform(current_trans)
object_final.paint_uniform_color([1, 0, 0])

# 计算误差指标
dists = np.asarray(object_final.compute_point_cloud_distance(scene_down))
valid_dists = dists[~np.isnan(dists)]

print("\n=== 配准结果验证 ===")
print(f"平均几何误差: {np.mean(valid_dists):.4f}m")
print(f"最大几何误差: {np.max(valid_dists):.4f}m")
print(f"RMSE: {np.sqrt(np.mean(valid_dists**2)):.4f}m")

# 可视化最终结果
print("\n最终配准效果：")
custom_draw_geometry([object_final, scene_down])

# 刚性变换验证
rotation = current_trans[:3, :3]
det = np.linalg.det(rotation)
print(f"\n旋转矩阵行列式: {det:.6f} (应接近1.0)")