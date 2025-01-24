# %% [markdown]
# # 3D物体点云匹配全流程
# 从预处理到结果输出的完整可交互流程

# %% [markdown]
# ## 1. 环境准备
# 安装必要库（首次运行需取消注释）

# %%
# !pip install open3d numpy matplotlib

# %% [markdown]
# ## 2. 初始设置
# 导入库并设置可视化参数

# %%
import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# 可视化配置
def custom_draw_geometry(pcd_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# %% [markdown]
# ## 3. 数据加载
# 显示原始点云的初始状态

# %%
# 加载点云
print("正在加载点云...")
object_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/cropped.ply")
scene_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/scene.ply")

# === 新增：沿XY平面翻转物体点云 ===
# 对Z坐标取反实现XY平面翻转
points = np.asarray(object_raw.points)
points[:, 2] *= -1  # 翻转Z轴坐标
object_raw.points = o3d.utility.Vector3dVector(points)

# 同时翻转法线方向（如果存在）
if object_raw.has_normals():
    normals = np.asarray(object_raw.normals)
    normals[:, 2] *= -1  # 保持法线方向一致性
    object_raw.normals = o3d.utility.Vector3dVector(normals)

# 为原始点云着色以便区分
object_raw.paint_uniform_color([1, 0, 0])  # 红色为物体
scene_raw.paint_uniform_color([0, 0, 1])   # 蓝色为场景

print("\n原始点云可视化：")
custom_draw_geometry([object_raw, scene_raw])

# 打印基本信息
print(f"\n物体点云点数: {len(object_raw.points)}")
print(f"场景点云点数: {len(scene_raw.points)}")
print("原始坐标系差异：")
print("物体中心:", object_raw.get_center())
print("场景中心:", scene_raw.get_center())

# %% [markdown]
# ## 4. 点云预处理（先执行）
# 降采样与去噪处理

# %%
def adaptive_downsample(pcd, target_points):
    """自适应降采样到目标点数"""
    current_points = len(pcd.points)
    if current_points <= target_points:
        return pcd
    
    # 计算初始体素尺寸
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()
    initial_voxel = np.cbrt((bbox_size[0]*bbox_size[1]*bbox_size[2])/target_points)
    
    # 二分法寻找合适体素尺寸
    low, high = 0.0, initial_voxel*2
    for _ in range(10):
        mid = (low + high) / 2
        down_pcd = pcd.voxel_down_sample(mid)
        if len(down_pcd.points) > target_points:
            low = mid
        else:
            high = mid
            
    return pcd.voxel_down_sample(high)

# 降采样处理
print("\n[预处理] 自适应降采样...")
object_down = adaptive_downsample(object_raw, 300000)  # 物体不超过30万点
scene_down = adaptive_downsample(scene_raw, 1000000)   # 场景不超过100万点

print(f"降采样后点数 | 物体: {len(object_down.points)}, 场景: {len(scene_down.points)}")

# 去噪处理
print("[预处理] 统计离群点去除...")
object_down, _ = object_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
scene_down, _ = scene_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# 法线估计（优化参数）
print("[预处理] 法线估计...")
search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
object_down.estimate_normals(search_param)
scene_down.estimate_normals(search_param)

# 可视化预处理结果
object_down.paint_uniform_color([1, 0, 0])
scene_down.paint_uniform_color([0, 0, 1])
print("\n预处理后点云：")
custom_draw_geometry([object_down, scene_down])
# %% [markdown]
# ## 5. 坐标系对齐（后执行）
# 将物体中心移至场景中心

# %%
# 深拷贝预处理后的数据
object_pcd = copy.deepcopy(object_down)
scene_pcd = copy.deepcopy(scene_down)

# 中心对齐（将物体中心移至场景中心）
print("\n[对齐] 中心对齐...")
object_center = object_pcd.get_center()
scene_center = scene_pcd.get_center()

# 计算平移向量并应用（修改关键点）
translation = scene_center - object_center  # 计算场景到物体的偏移量
object_pcd.translate(translation)           # 移动物体到场景中心

# 可视化对齐结果
print("\n对齐后点云：")
object_pcd.paint_uniform_color([1, 0, 0])  # 保持物体为红色
scene_pcd.paint_uniform_color([0, 0, 1])   # 场景保持蓝色
custom_draw_geometry([object_pcd, scene_pcd])

# %% [markdown]
# ## 6. 直接ICP配准
# 基于对齐后的点云进行配准

# %%
# 设置ICP参数
print("\n开始ICP配准...")
max_correspondence_distance = 10  # 初始较大搜索距离
icp_iterations = [100, 50, 30]     # 多阶段迭代

current_trans = np.identity(4)

# 多阶段ICP
for i, max_iter in enumerate(icp_iterations):
    print(f"阶段 {i+1}: 最大迭代 {max_iter} 次")
    result = o3d.pipelines.registration.registration_icp(
        object_pcd, scene_pcd,  # 使用对齐后的点云
        max_correspondence_distance,
        current_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iter)
    )
    current_trans = result.transformation
    max_correspondence_distance *= 0.5  # 逐步收紧搜索范围

# 显示最终结果
object_final = copy.deepcopy(object_pcd)
object_final.transform(current_trans)
object_final.paint_uniform_color([1, 0, 0])
print("\n配准结果：")
custom_draw_geometry([scene_pcd, object_final])

# %% [markdown]
# ## 7. 结果输出与分析

# %%
# 转换矩阵分解
T = current_trans
translation_vector = T[:3, 3]

# 计算配准误差
dists = object_pcd.compute_point_cloud_distance(scene_pcd)  # 使用对齐后的点云
dist_array = np.asarray(dists)

# 输出参数
print("\n最终变换矩阵：")
print(T)
print(f"\n平移向量 (米): {translation_vector}")
print(f"平均配准误差: {np.mean(dist_array):.4f} 米")
print(f"最大配准误差: {np.max(dist_array):.4f} 米")

# 误差可视化
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.hist(dist_array, bins=50, range=[0, np.percentile(dist_array, 95)])
plt.title("配准误差分布")

plt.subplot(122)
plt.scatter(range(len(dist_array)), dist_array, s=1, alpha=0.3)
plt.ylim(0, np.percentile(dist_array, 95))
plt.title("逐点误差分布")
plt.show()

# 3D误差热力图
colors = plt.cm.jet(dist_array / np.percentile(dist_array, 95))
object_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
print("\n三维误差热力图（红色表示误差大）：")
custom_draw_geometry([object_pcd, scene_pcd])