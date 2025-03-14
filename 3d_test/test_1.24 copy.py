# %% [markdown]
# ## 点云配准流程（增强可视化版）
# 本 Notebook 实现改进的点云配准流程，包含法线可视化与多阶段误差分析

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
def custom_draw_geometry(pcd_list, show_normals=False, normal_scale=0.5):
    """增强可视化函数，支持法线显示"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 添加几何体
    for geom in pcd_list:
        vis.add_geometry(geom)
    
    # 设置渲染参数
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().line_width = 1.0
    vis.run()
    vis.destroy_window()

def visualize_normals(pcd, normal_scale=0.5):
    """法线方向可视化专用函数（兼容新版Open3D）"""
    # 获取点和法线数据
    points = np.asarray(pcd.points)
    if not pcd.has_normals() or len(points) == 0:
        return o3d.geometry.LineSet()
    
    normals = np.asarray(pcd.normals)
    
    # 创建线段几何体
    lines = []
    line_points = []
    
    for i in range(points.shape[0]):
        # 起点
        start = points[i]
        # 终点（沿法线方向延伸）
        end = start + normals[i] * normal_scale
        
        line_points.append(start)
        line_points.append(end)
        lines.append([2*i, 2*i+1])  # 线段索引
    
    # 创建LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.paint_uniform_color([0, 1, 0])  # 绿色
    
    return line_set

# %% [markdown]
# ### 3. 数据加载与初始化

# %%
# 加载点云（注意修改文件路径）
print("正在加载点云...")
object_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/cropped_1.ply")
scene_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/scene_cropped_1.ply")

# 初始位置调整
translation = - scene_raw.get_center() + object_raw.get_center()
object_raw.translate(translation * 0.1)

# 颜色标记
object_raw.paint_uniform_color([1, 0, 0])  # 红色为物体
scene_raw.paint_uniform_color([0, 0, 1])   # 蓝色为场景

print("\n初始位置可视化：")
custom_draw_geometry([object_raw, scene_raw])

# %% [markdown]
# ### 4. 数据预处理（含法线估计）

# %%
# 自适应降采样函数
def adaptive_downsample(pcd, target_points):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    volume = np.prod(extent)
    
    initial_voxel = np.cbrt(volume / max(target_points, 100))
    low, high = 0.0, initial_voxel * 3
    best_pcd = pcd
    
    for _ in range(15):
        mid = (low + high) / 2
        down_pcd = pcd.voxel_down_sample(mid)
        if len(down_pcd.points) > target_points * 1.2:
            low = mid
        else:
            high = mid
            if len(down_pcd.points) > 100:
                best_pcd = down_pcd
    return best_pcd

# 执行预处理流程
'''
print("\n[预处理] 自适应降采样...")
object_down = adaptive_downsample(object_raw, 400)
scene_down = adaptive_downsample(scene_raw, 3000)
print(f"降采样后点数 | 物体: {len(object_down.points)}, 场景: {len(scene_down.points)}")
'''
object_down = object_raw
scene_down = scene_raw
# 去噪处理
print("[预处理] 统计离群点去除...")
object_down, _ = object_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
scene_down, _ = scene_down.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.8)

# 法线估计与方向统一
print("[预处理] 法线估计...")
object_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
scene_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=80))
'''
object_down.orient_normals_to_align_with_direction([0, 0, 1])
scene_down.orient_normals_to_align_with_direction([0, 0, 1])
'''

# 可视化预处理结果（带法线）
temp_object = copy.deepcopy(object_down).paint_uniform_color([1,0,0])
temp_scene = copy.deepcopy(scene_down).paint_uniform_color([0,0,1])
print("\n预处理后点云及法线分布：")
custom_draw_geometry([temp_object, 
                     temp_scene,
                     visualize_normals(temp_object, 1),
                     visualize_normals(temp_scene, 1)])

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
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(20000, 0.999)
)

print(f"粗配准得分: {ransac_result.fitness:.3f}")

# 可视化粗配准结果（带法线）
object_ransac = copy.deepcopy(object_down_ransac).transform(ransac_result.transformation)
object_ransac.paint_uniform_color([1,0,0])
print("\n粗配准结果（带法线方向）：")
custom_draw_geometry([object_ransac, scene_down_ransac, 
                     visualize_normals(object_ransac), 
                     visualize_normals(scene_down_ransac)], show_normals=True)

# %% [markdown]
# ### 6. 多阶段ICP精配准

# %%
# ICP阶段参数设置
icp_stages = [
    {"max_iter": 100, "max_dist": 2.0},
    {"max_iter": 200, "max_dist": 1.0},
    {"max_iter": 300, "max_dist": 0.5},
    {"max_iter": 500, "max_dist": 0.2}
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

# 计算几何误差
dists = object_final.compute_point_cloud_distance(scene_down)
dist_array = np.asarray(dists)
valid_dists = dist_array[~np.isnan(dist_array)]

# 法线一致性分析
print("\n=== 法线一致性分析 ===")
final_normals = np.asarray(object_final.normals)
scene_normals = np.asarray(scene_down.normals)

scene_tree = o3d.geometry.KDTreeFlann(scene_down)
angle_errors = []
for i in range(len(final_normals)):
    [k, idx, _] = scene_tree.search_knn_vector_3d(object_final.points[i], 1)
    cos_theta = np.dot(final_normals[i], scene_normals[idx[0]])
    angle_errors.append(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

angle_errors = np.rad2deg(np.array(angle_errors))  # 转换为角度
print(f"平均法线偏差: {np.mean(angle_errors):.2f}°")
print(f"最大法线偏差: {np.max(angle_errors):.2f}°")

# 可视化设置
max_error = np.percentile(valid_dists, 95)
max_angle = np.percentile(angle_errors, 95)

# 几何误差可视化
dist_colors = plt.cm.jet(np.clip(valid_dists/max_error, 0, 1))[:, :3]
dist_pcd = copy.deepcopy(object_final)
dist_pcd.colors = o3d.utility.Vector3dVector(dist_colors)

# 法线误差可视化
angle_colors = plt.cm.jet(angle_errors / max_angle)[:, :3]
normal_error_pcd = copy.deepcopy(object_final)
normal_error_pcd.colors = o3d.utility.Vector3dVector(angle_colors)

# 结果展示
print("\n=== 综合验证结果 ===")
print("几何误差分布（红色表示误差大）：")
custom_draw_geometry([dist_pcd, scene_down])

print("\n法线偏差分布（红色表示偏差大）：")
custom_draw_geometry([normal_error_pcd, scene_down, 
                     visualize_normals(normal_error_pcd, 0.1),
                     visualize_normals(scene_down, 0.1)], show_normals=True)

# 刚性变换验证
rotation = current_trans[:3, :3]
det = np.linalg.det(rotation)
print(f"\n旋转矩阵行列式: {det:.4f} (应接近1.0)")

# %% [markdown]
# ### 8. 配准轨迹可视化

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

print("\n配准轨迹可视化（含法线方向）：")
custom_draw_geometry([scene_down, normal_error_pcd] + trajectory + 
                    [visualize_normals(normal_error_pcd, 0.1),
                     visualize_normals(scene_down, 0.1)], show_normals=True)