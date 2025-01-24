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

# 加载点云
print("正在加载点云...")
object_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/cropped.ply")
scene_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/scene.ply")

translation = scene_raw.get_center() - object_raw.get_center()
object_raw.translate(translation)

# 原始点云着色
object_raw.paint_uniform_color([1, 0, 0])
scene_raw.paint_uniform_color([0, 0, 1])

print("\n原始点云可视化：")
custom_draw_geometry([object_raw, scene_raw])

# 自适应降采样函数
def adaptive_downsample(pcd, target_points):
    current_points = len(pcd.points)
    if current_points <= target_points:
        return pcd
    
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()
    initial_voxel = np.cbrt((bbox_size[0]*bbox_size[1]*bbox_size[2])/target_points)
    
    low, high = 0.0, initial_voxel*2
    for _ in range(10):
        mid = (low + high)/2
        down_pcd = pcd.voxel_down_sample(mid)
        if len(down_pcd.points) > target_points:
            low = mid
        else:
            high = mid
            
    return pcd.voxel_down_sample(high)

# === 预处理流程 ===
print("\n[预处理] 自适应降采样...")
object_down = adaptive_downsample(object_raw, 30000)
scene_down = adaptive_downsample(scene_raw, 100000)

print("[预处理] 统计离群点去除...")
object_down, _ = object_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
scene_down, _ = scene_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

print("[预处理] 法线估计与方向统一...")
search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
object_down.estimate_normals(search_param)
scene_down.estimate_normals(search_param)

# 统一法线方向
object_down.orient_normals_to_align_with_direction([0, 0, 1])
scene_down.orient_normals_to_align_with_direction([0, 0, 1])

object_down.paint_uniform_color([1, 0, 0])
scene_down.paint_uniform_color([0, 0, 1])
print("\n预处理后点云：")
custom_draw_geometry([object_down, scene_down])

# === 坐标系对齐 ===
object_pcd = copy.deepcopy(object_down)
scene_pcd = copy.deepcopy(scene_down)

print("\n[对齐] 中心对齐...")
translation = scene_pcd.get_center() - object_pcd.get_center()
object_pcd.translate(translation)

# === 粗配准阶段 ===
print("\n[阶段0] RANSAC粗配准...")
def extract_fpfh(pcd, radius=0.1):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )

object_fpfh = extract_fpfh(object_pcd)
scene_fpfh = extract_fpfh(scene_pcd)

ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    object_pcd, scene_pcd,
    object_fpfh, scene_fpfh,
    mutual_filter=True,
    max_correspondence_distance=5.0,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(5.0)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
)
current_trans = ransac_result.transformation
print(f"粗配准得分: {ransac_result.fitness:.3f}")

# === 多阶段ICP配准 ===
print("\n[阶段1-4] 多阶段ICP配准...")
icp_stages = [
    {"max_iter": 200, "max_dist": 10.0},  # 粗配准
    {"max_iter": 100, "max_dist": 5.0},   # 中等精度
    {"max_iter": 50, "max_dist": 2.0},    # 精细配准
    {"max_iter": 30, "max_dist": 0.5}     # 最终优化
]

for stage_idx, stage in enumerate(icp_stages):
    print(f"阶段 {stage_idx+1}: {stage['max_iter']}次迭代 | 搜索距离 {stage['max_dist']}米")
    
    result = o3d.pipelines.registration.registration_icp(
        object_pcd, scene_pcd,
        stage["max_dist"],
        current_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7,
            relative_rmse=1e-7,
            max_iteration=stage["max_iter"]
        )
    )
    current_trans = result.transformation
    print(f"阶段{stage_idx+1} 匹配点数: {len(result.correspondence_set)}")

# === 最终结果 ===
object_final = copy.deepcopy(object_pcd)
object_final.transform(current_trans)

# === 结果分析 ===
print("\n=== 配准结果 ===")
# 转换矩阵分解
T = current_trans
translation_vector = T[:3, 3]
rotation_matrix = T[:3, :3]

# 计算配准误差
dists = object_final.compute_point_cloud_distance(scene_pcd)
dist_array = np.asarray(dists)
valid_dists = dist_array[~np.isnan(dist_array)]

# 误差指标
print(f"平均误差: {np.mean(valid_dists):.4f} m")
print(f"最大误差: {np.max(valid_dists):.4f} m")
print(f"RMSE: {np.sqrt(np.mean(valid_dists**2)):.4f} m")

# 重叠率计算
overlap_ratio = o3d.pipelines.registration.evaluate_registration(
    object_final, scene_pcd, 0.05
).fitness
print(f"重叠率: {overlap_ratio*100:.1f}%")

# 刚性变换验证
det = np.linalg.det(rotation_matrix)
print(f"旋转矩阵行列式: {det:.4f} (应为接近1.0)")

# === 可视化 ===
# 误差热力图
max_display_error = np.percentile(valid_dists, 95)
colors = plt.cm.jet(np.clip(valid_dists/max_display_error, 0, 1))[:, :3]

error_pcd = copy.deepcopy(object_final)
error_pcd.colors = o3d.utility.Vector3dVector(colors)

print("\n最终配准结果：")
custom_draw_geometry([error_pcd, scene_pcd])

# 误差分布图
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.hist(valid_dists, bins=50, density=True)
plt.title("误差分布直方图")

plt.subplot(132)
plt.boxplot(valid_dists, vert=False)
plt.title("误差箱线图")

plt.subplot(133)
plt.scatter(range(len(valid_dists)), valid_dists, s=1, alpha=0.3)
plt.ylim(0, max_display_error)
plt.title("逐点误差分布")
plt.tight_layout()
plt.show()