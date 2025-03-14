# %% [markdown]
# ## 点云配准流程（Jupyter版）
# 本 Notebook 实现了一个改进的点云配准流程，包含预处理、多阶段配准和结果验证。

# %% [markdown]
# ### 1. 导入依赖库

# %%
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

# %% [markdown]
# ### 2. 定义可视化函数

# %%
def custom_draw_geometry(pcd_list):
    """改进的可视化函数，支持多窗口显示"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# %% [markdown]
# ### 3. 数据加载与初始化

# %%
# 加载点云（注意修改文件路径）
print("正在加载点云...")
object_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/cropped.ply")
scene_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/scene.ply")

# 初始位置调整
translation = scene_raw.get_center() - object_raw.get_center()
object_raw.translate(translation * 0.8)

# 颜色标记
object_raw.paint_uniform_color([1, 0, 0])  # 红色为物体
scene_raw.paint_uniform_color([0, 0, 1])   # 蓝色为场景

print("\n初始位置可视化：")
custom_draw_geometry([object_raw, scene_raw])

# %% [markdown]
# ### 4. 数据预处理

# %%
# 自适应降采样函数
def adaptive_downsample(pcd, target_points):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    volume = np.prod(extent)
    
    initial_voxel = np.cbrt(volume / max(target_points, 1000))
    low, high = 0.0, initial_voxel * 3
    best_pcd = pcd
    
    for _ in range(15):
        mid = (low + high) / 2
        down_pcd = pcd.voxel_down_sample(mid)
        if len(down_pcd.points) > target_points * 1.2:
            low = mid
        else:
            high = mid
            if len(down_pcd.points) > 1000:
                best_pcd = down_pcd
    return best_pcd

# %%
# 执行预处理流程
print("\n[预处理] 自适应降采样...")
object_down = adaptive_downsample(object_raw, 30000)
scene_down = adaptive_downsample(scene_raw, 100000)
print(f"降采样后点数 | 物体: {len(object_down.points)}, 场景: {len(scene_down.points)}")

# 去噪处理
print("[预处理] 统计离群点去除...")
object_down, _ = object_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
scene_down, _ = scene_down.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.8)

# 法线估计
print("[预处理] 法线估计...")
object_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
scene_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=80))

# 统一法线方向
object_down.orient_normals_to_align_with_direction([0, 0, 1])
scene_down.orient_normals_to_align_with_direction([0, 0, 1])

# 可视化预处理结果
temp_object = copy.deepcopy(object_down).paint_uniform_color([1,0,0])
temp_scene = copy.deepcopy(scene_down).paint_uniform_color([0,0,1])
print("\n预处理后点云分布：")
custom_draw_geometry([temp_object, temp_scene])

# %% [markdown]
# ### 5. 特征提取与粗配准

# %%
# FPFH特征提取函数
def compute_fpfh(pcd, voxel_size):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4,
            max_nn=150
        )
    )

# 执行RANSAC粗配准
print("\n[阶段1] RANSAC粗配准...")
voxel_size = 0.1

# 降采样用于RANSAC
object_down_ransac = object_down.voxel_down_sample(voxel_size)
scene_down_ransac = scene_down.voxel_down_sample(voxel_size)

# 计算特征
object_fpfh = compute_fpfh(object_down_ransac, voxel_size)
scene_fpfh = compute_fpfh(scene_down_ransac, voxel_size)

# 执行RANSAC
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    object_down_ransac, scene_down_ransac,
    object_fpfh, scene_fpfh,
    mutual_filter=False,
    max_correspondence_distance=voxel_size * 3,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    ransac_n=3,
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999)
)

print(f"粗配准得分: {ransac_result.fitness:.3f}")

# 可视化粗配准结果
object_ransac = copy.deepcopy(object_down_ransac).transform(ransac_result.transformation)
object_ransac.paint_uniform_color([1,0,0])
print("\n粗配准结果：")
custom_draw_geometry([object_ransac, scene_down_ransac])

# %% [markdown]
# ### 6. 多阶段ICP精配准

# %%
# ICP阶段参数设置
icp_stages = [
    {"max_iter": 100, "max_dist": 5.0},
    {"max_iter": 200, "max_dist": 2.0},
    {"max_iter": 300, "max_dist": 0.8},
    {"max_iter": 500, "max_dist": 0.3}
]

current_trans = ransac_result.transformation

for stage_idx, stage in enumerate(icp_stages):
    print(f"\n阶段 {stage_idx+2}: {stage['max_iter']}次迭代 | 搜索距离 {stage['max_dist']}米")
    
    result = o3d.pipelines.registration.registration_icp(
        object_down, scene_down,
        stage["max_dist"],
        current_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=stage["max_iter"],
            relative_fitness=1e-8,
            relative_rmse=1e-8
        )
    )
    current_trans = result.transformation
    print(f"匹配点数: {len(result.correspondence_set)} | 当前RMSE: {result.inlier_rmse:.4f}")

# %% [markdown]
# ### 7. 结果验证与可视化

# %%
# 最终变换应用
object_final = copy.deepcopy(object_down).transform(current_trans)

# 计算误差指标
dists = object_final.compute_point_cloud_distance(scene_down)
dist_array = np.asarray(dists)
valid_dists = dist_array[~np.isnan(dist_array)]

print("\n=== 配准验证 ===")
print(f"平均误差: {np.mean(valid_dists):.4f}m")
print(f"RMSE: {np.sqrt(np.mean(valid_dists**2)):.4f}m")

# 刚性变换验证
rotation = current_trans[:3, :3]
det = np.linalg.det(rotation)
print(f"旋转矩阵行列式: {det:.4f} (应接近1.0)")

# 误差热力图
max_error = np.percentile(valid_dists, 95)
colors = plt.cm.jet(np.clip(valid_dists/max_error, 0, 1))[:, :3]
error_pcd = copy.deepcopy(object_final)
error_pcd.colors = o3d.utility.Vector3dVector(colors)

print("\n最终配准效果（红色为误差分布）：")
custom_draw_geometry([error_pcd, scene_down])

# %% [markdown]
# ### 8. 配准轨迹可视化（可选）

# %%
# 随机采样100个点显示配准轨迹
trajectory = []
for idx in np.random.choice(len(object_down.points), 100, replace=False):
    src = np.asarray(object_down.points)[idx]
    dst = np.asarray(object_final.points)[idx]
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([src, dst])
    line.lines = [[0, 1]]
    line.colors = o3d.utility.Vector3dVector([[1,0,0]])
    trajectory.append(line)

print("\n配准轨迹可视化：")
custom_draw_geometry([scene_down, error_pcd] + trajectory)