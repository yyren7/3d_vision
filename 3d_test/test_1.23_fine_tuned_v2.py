import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

def custom_draw_geometry(pcd_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# 加载点云
print("正在加载点云...")
object_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/cropped.ply")
scene_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/scene.ply")

# === 改进1：调整初始位置 ===
# 先进行粗略中心对齐（保留部分偏移）
translation = scene_raw.get_center() - object_raw.get_center()
object_raw.translate(translation * 0.8)  # 保留20%偏移供配准算法处理

# 原始点云着色
object_raw.paint_uniform_color([1, 0, 0])
scene_raw.paint_uniform_color([0, 0, 1])

print("\n初始位置可视化：")
custom_draw_geometry([object_raw, scene_raw])

# === 改进2：优化降采样参数 ===
def adaptive_downsample(pcd, target_points):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    volume = np.prod(extent)
    
    # 动态调整体素尺寸
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

print("\n[预处理] 自适应降采样...")
object_down = adaptive_downsample(object_raw, 30000)
scene_down = adaptive_downsample(scene_raw, 100000)
print(f"降采样后点数 | 物体: {len(object_down.points)}, 场景: {len(scene_down.points)}")

# === 改进3：增强去噪处理 ===
print("[预处理] 统计离群点去除...")
object_down, _ = object_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
scene_down, _ = scene_down.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.8)

# === 改进4：优化法线估计 ===
print("[预处理] 法线估计...")
object_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
scene_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=80))

# 统一法线方向（根据场景调整）
object_down.orient_normals_to_align_with_direction([0, 0, 1])
scene_down.orient_normals_to_align_with_direction([0, 0, 1])

# === 改进5：添加预处理可视化 ===
temp_object = copy.deepcopy(object_down).paint_uniform_color([1,0,0])
temp_scene = copy.deepcopy(scene_down).paint_uniform_color([0,0,1])
print("\n预处理后点云分布：")
custom_draw_geometry([temp_object, temp_scene])

# === 改进6：优化特征提取 ===
def compute_fpfh(pcd, voxel_size):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4,  # 扩大特征搜索范围
            max_nn=150
        )
    )

# === 改进7：分阶段配准流程 ===
# 阶段1：粗配准（RANSAC）
print("\n[阶段1] RANSAC粗配准...")
voxel_size = 0.1  # 根据点云密度调整

object_down_ransac = object_down.voxel_down_sample(voxel_size)
scene_down_ransac = scene_down.voxel_down_sample(voxel_size)

object_fpfh = compute_fpfh(object_down_ransac, voxel_size)
scene_fpfh = compute_fpfh(scene_down_ransac, voxel_size)

ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    object_down_ransac, scene_down_ransac,
    object_fpfh, scene_fpfh,
    mutual_filter=False,  # 关闭互滤增加候选点
    max_correspondence_distance=voxel_size * 3,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    ransac_n=3,  # 减少最小采样点
    checkers=[],  # 暂时关闭检查器
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
        200000,  # 增加最大迭代次数
        0.999
    )
)
current_trans = ransac_result.transformation
print(f"粗配准得分: {ransac_result.fitness:.3f}")

# 可视化粗配准结果
object_ransac = copy.deepcopy(object_down_ransac).transform(ransac_result.transformation)
object_ransac.paint_uniform_color([1,0,0])
print("\n粗配准结果：")
custom_draw_geometry([object_ransac, scene_down_ransac])

# 阶段2-5：多阶段ICP
print("\n[阶段2-5] 多阶段ICP优化...")
icp_stages = [
    {"max_iter": 100, "max_dist": 5.0},  # 大范围搜索
    {"max_iter": 200, "max_dist": 2.0},  # 中等精度
    {"max_iter": 300, "max_dist": 0.8},  # 精细配准
    {"max_iter": 500, "max_dist": 0.3}   # 最终优化
]

for stage_idx, stage in enumerate(icp_stages):
    print(f"阶段 {stage_idx+2}: {stage['max_iter']}次迭代 | 搜索距离 {stage['max_dist']}米")
    
    result = o3d.pipelines.registration.registration_icp(
        object_down, scene_down,
        stage["max_dist"],
        current_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=stage["max_iter"],
            relative_fitness=1e-8,  # 更严格的收敛条件
            relative_rmse=1e-8
        )
    )
    current_trans = result.transformation
    print(f"匹配点数: {len(result.correspondence_set)} | 当前RMSE: {result.inlier_rmse:.4f}")

# === 最终结果处理 ===
object_final = copy.deepcopy(object_down)
object_final.transform(current_trans)

# === 结果验证 ===
print("\n=== 配准验证 ===")
# 计算误差指标
dists = object_final.compute_point_cloud_distance(scene_down)
dist_array = np.asarray(dists)
valid_dists = dist_array[~np.isnan(dist_array)]

if len(valid_dists) == 0:
    raise ValueError("无有效对应点，请检查配准结果")

print(f"平均误差: {np.mean(valid_dists):.4f}m")
print(f"RMSE: {np.sqrt(np.mean(valid_dists**2)):.4f}m")

# 刚性变换验证
rotation = current_trans[:3, :3]
det = np.linalg.det(rotation)
if not (0.9 < det < 1.1):
    print(f"警告：非刚性变换 (行列式={det:.4f})")

# === 可视化增强 ===
# 误差热力图
max_error = np.percentile(valid_dists, 95)
colors = plt.cm.jet(np.clip(valid_dists/max_error, 0, 1))[:, :3]

error_pcd = copy.deepcopy(object_final)
error_pcd.colors = o3d.utility.Vector3dVector(colors)

print("\n最终配准效果：")
custom_draw_geometry([error_pcd, scene_down])

# 轨迹可视化
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

