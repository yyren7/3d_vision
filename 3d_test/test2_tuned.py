import open3d as o3d
import numpy as np
import math


def load_point_cloud(file_path):
    """
    读取点云文件（包括 .ply, .pcd, .xyz, .obj 等），并返回 open3d.geometry.PointCloud 对象。
    如果点云为空或文件无效会报错。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"无法读取文件 {file_path} 或点云为空。")
    return pcd


def preprocess_point_cloud(pcd, voxel_size=0.01):
    """
    对点云进行体素下采样，并估计法线，随后计算 FPFH 特征。

    :param pcd: 输入点云
    :param voxel_size: 体素下采样大小（单位：米，根据你实际场景调整）
    :return: (pcd_down, fpfh) 下采样后的点云、以及计算好的 FPFH 特征
    """
    # 1) 下采样
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 2) 法线估计
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )
    # 视需求可执行normalize_normals()，也可保留默认

    # 3) 计算 FPFH 特征
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100
        )
    )

    return pcd_down, fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,
                                distance_threshold=0.05):
    """
    基于 FPFH 特征的全局 RANSAC 匹配，获取初始变换矩阵。

    :param source_down: 下采样后的源点云（物体/模板）
    :param target_down: 下采样后的目标点云（场景/环境）
    :param source_fpfh: 源点云的 FPFH 特征
    :param target_fpfh: 目标点云的 FPFH 特征
    :param distance_threshold: RANSAC 拟合的最大对应点距离阈值
    :return: RANSAC 的配准结果 (o3d.pipelines.registration.RegistrationResult)
    """
    # 构建对应关系检查器
    checker = [
        # 几何一致性检查 (边长)
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        # 距离检查
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ]

    # 使用 RANSAC 进行全局配准
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=checker,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result_ransac


def refine_registration(source, target, initial_transform, distance_threshold=0.02):
    """
    使用 ICP 做精细配准，得到最终变换矩阵。

    :param source: 源点云（可以是不下采样或较少下采样的原始点云）
    :param target: 目标点云（可以是不下采样或较少下采样的原始点云）
    :param initial_transform: 初始变换矩阵 (4x4)
    :param distance_threshold: ICP 允许的最大对应点距离
    :return: ICP 的配准结果 (o3d.pipelines.registration.RegistrationResult)
    """
    # 使用 PointToPlane ICP（如果法线信息质量足够好）
    result_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp


def rotation_matrix_to_euler_angles(R):
    """
    将 3x3 旋转矩阵转换为 (roll, pitch, yaw) 欧拉角。
    下面以 XYZ (roll->pitch->yaw) 为例；如有需要可自行修改为 ZYX 或其他顺序。

    返回值为弧度制的 (roll, pitch, yaw)
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])  # X
        pitch = math.atan2(-R[2, 0], sy)  # Y
        yaw = math.atan2(R[1, 0], R[0, 0])  # Z
    else:
        # 处理奇异状态
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw


def get_6d_pose_from_transformation(transformation):
    """
    从 4x4 齐次变换矩阵中提取 (x, y, z, roll, pitch, yaw)。
    """
    x, y, z = transformation[0, 3], transformation[1, 3], transformation[2, 3]
    R = transformation[:3, :3]
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
    return x, y, z, roll, pitch, yaw


def visualize_registration(source, target, transformation=np.eye(4)):
    """
    可视化配准结果：将源点云变换后，与目标点云一起显示。
    """
    source_temp = source.clone()
    source_temp.transform(transformation)
    # 分配可视化用的颜色
    source_temp.paint_uniform_color([1, 0, 0])  # 源点云红色
    target_temp = target.clone()
    target_temp.paint_uniform_color([0, 1, 0])  # 目标点云绿色

    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      width=800,
                                      height=600,
                                      window_name="Registration Visualization")


def main():
    # === 1. 设置文件路径（请根据实际文件名修改）===
    template_ply_path = "../resources/3d/output1.ply"  # 物体(模板)点云, binary_little_endian + double
    scene_ply_path = "../resources/3d/RGBDPoints_1733466519911547.ply"  # 环境(场景)点云, ascii + float

    # === 2. 读取点云 ===
    print("读取点云...")
    source_raw = load_point_cloud(template_ply_path)  # 源点云（需要找的物体）
    target_raw = load_point_cloud(scene_ply_path)  # 目标点云（环境）

    # === 3. 预处理：下采样 + 法线估计 + FPFH 特征计算 ===
    voxel_size = 0.01  # 根据你场景尺度可适当修改
    print("下采样与特征提取...")
    source_down, source_fpfh = preprocess_point_cloud(source_raw, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_raw, voxel_size)

    # === 4. 全局 RANSAC 配准，获得初始变换 ===
    distance_threshold = voxel_size * 5.0
    print("执行全局 RANSAC...")
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh,
        distance_threshold=distance_threshold
    )
    print("RANSAC 初始变换：\n", result_ransac.transformation)

    # === 5. 基于初始变换的 ICP 精细配准 ===
    print("执行 ICP 精细配准...")
    result_icp = refine_registration(
        source_raw,  # 使用原始点云可得到更精确的结果
        target_raw,
        result_ransac.transformation,
        distance_threshold=distance_threshold / 2.0
    )
    final_transformation = result_icp.transformation
    print("ICP 精细配准后的变换矩阵：\n", final_transformation)

    # === 6. 提取最终 6D 位姿 (x, y, z, roll, pitch, yaw) ===
    x, y, z, roll, pitch, yaw = get_6d_pose_from_transformation(final_transformation)
    print("\n========== 最终匹配结果 ==========")
    print(f"平移：x = {x:.4f}, y = {y:.4f}, z = {z:.4f}")
    print(f"旋转（欧拉角，单位：弧度）：roll = {roll:.4f}, pitch = {pitch:.4f}, yaw = {yaw:.4f}")
    print("=================================")

    # === 7. 可视化（可选）===
    # 如果要查看可视化，请解除下面的注释
    # visualize_registration(source_raw, target_raw, final_transformation)


if __name__ == "__main__":
    main()
