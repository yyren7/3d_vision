# %% [markdown]
# ## 点云粗配准流程
# 本 Notebook 实现基于RANSAC的特征匹配粗配准

# %% [markdown]
# ### 1. 导入依赖库

# %%
import open3d as o3d
import numpy as np
import copy

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
# 加载点云
print("正在加载点云...")
object_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/output1.ply")
scene_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/scene_cropped_1.ply")
scale_factor = 0.01  # 假设原始数据单位为厘米，转换为米

object_raw.scale(scale_factor, center=(0,0,0))
scene_raw.scale(scale_factor, center=(0,0,0))
# 初始位置调整
translation = scene_raw.get_center() - object_raw.get_center()
object_raw.translate(translation * 0.5)

# 颜色标记
object_colored = copy.deepcopy(object_raw).paint_uniform_color([1, 0, 0])
scene_colored = copy.deepcopy(scene_raw).paint_uniform_color([0, 0, 1])

print("\n初始位置可视化：")
custom_draw_geometry([object_colored, scene_colored])

# %% [markdown]
# ### 4. 数据预处理

# %%
# 降采样参数
voxel_size = 0.025  # 减小体素大小以保留更多细节

print("\n[预处理] 降采样...")
object_down = object_raw.voxel_down_sample(voxel_size)
scene_down = scene_raw.voxel_down_sample(voxel_size)

# 去噪处理
print("[预处理] 统计离群点去除...")
object_down, _ = object_down.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.8)
scene_down, _ = scene_down.remove_statistical_outlier(nb_neighbors=55, std_ratio=2.0)

# 法线估计
print("[预处理] 法线估计...")
object_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=40))
scene_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=40))

print("\n预处理可视化：")
custom_draw_geometry([object_down, scene_down])
# %% [markdown]
# ### 5. 特征提取与粗配准

# %%
# FPFH特征提取
def compute_fpfh(pcd, voxel_size):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 7,  # 增加搜索半径
            max_nn=150  # 增加最大邻居点数
        )
    )

print("\n[特征提取] 计算FPFH特征...")
object_fpfh = compute_fpfh(object_down, voxel_size)
scene_fpfh = compute_fpfh(scene_down, voxel_size)

# 执行RANSAC粗配准
print("\n[粗配准] 执行RANSAC...")
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    object_down, 
    scene_down,
    object_fpfh, 
    scene_fpfh,
    mutual_filter=True,
    max_correspondence_distance=voxel_size * 5,  # 增加对应距离阈值
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),  # 使用严格点对点估计
    ransac_n=5,  # 增加用于估计的点数
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 5)
    ],  # 添加对应检查器
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999)  # 增加迭代次数和置信度
)

# 应用变换矩阵
object_registered = copy.deepcopy(object_down).transform(ransac_result.transformation)
object_registered.paint_uniform_color([1, 0, 0])

print("\n粗配准结果可视化：")
custom_draw_geometry([object_registered, scene_down])

# %% [markdown]
# ### 6. 配准结果输出

# %%
print("\n=== 粗配准结果 ===")
print(f"配准得分: {ransac_result.fitness:.3f}")
print(f"变换矩阵:\n{ransac_result.transformation}")

# 在粗配准之后添加ICP精配准
print("\n[精配准] 执行ICP...")
icp_result = o3d.pipelines.registration.registration_icp(
    object_down, scene_down, 
    voxel_size * 1.5,  # 对应距离阈值
    ransac_result.transformation,  # 使用粗配准结果作为初始值
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
)

# 应用ICP变换矩阵
object_icp = copy.deepcopy(object_down).transform(icp_result.transformation)
object_icp.paint_uniform_color([1, 0.5, 0])  # 使用橙色区分ICP结果

print("\n精配准结果可视化：")
custom_draw_geometry([object_icp, scene_down])

print("\n=== 精配准结果 ===")
print(f"精配准得分: {icp_result.fitness:.3f}")
print(f"精配准RMSE: {icp_result.inlier_rmse:.5f}")
print(f"精配准变换矩阵:\n{icp_result.transformation}")

# 显示粗配准和精配准的对比
print("\n粗配准与精配准对比：")
object_ransac = copy.deepcopy(object_down).transform(ransac_result.transformation)
object_ransac.paint_uniform_color([1, 0, 0])  # 红色表示粗配准
object_icp.paint_uniform_color([0, 1, 0])     # 绿色表示精配准
custom_draw_geometry([object_ransac, object_icp, scene_down])