import open3d as o3d
import numpy as np
import time

# 读取点云
source = o3d.io.read_point_cloud("../output1.ply")
target = o3d.io.read_point_cloud("../RGBDPoints_1733466519911547.ply")

# 体素下采样（减少点云密度，加速计算）
voxel_size = 0.05
source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)

# 估算法向量
print("估算法向量...")
source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

# 计算 FPFH 特征
def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2  # 法向量估算的搜索半径
    radius_feature = voxel_size * 5  # FPFH 特征计算的搜索半径
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

print("计算 FPFH 特征...")
source_fpfh = compute_fpfh(source_down, voxel_size)
target_fpfh = compute_fpfh(target_down, voxel_size)

# 全局粗配准：RANSAC
print("开始全局粗配准...")
distance_threshold = voxel_size * 1.5
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, True,
    distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999))

print("RANSAC 变换矩阵:\n", result_ransac.transformation)

# 使用 RANSAC 结果作为初始变换执行 ICP
print("开始精细 ICP...")
max_correspondence_distance = voxel_size * 0.4
result_icp = o3d.pipelines.registration.registration_icp(
    source_down, target_down, max_correspondence_distance, result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))

print("ICP 优化后的变换矩阵:\n", result_icp.transformation)

# 应用变换
source_down.transform(result_icp.transformation)

# 可视化结果
o3d.visualization.draw_geometries([source_down, target_down])
