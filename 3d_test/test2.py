import open3d as o3d
import numpy as np
import math


def load_point_cloud(file_path):
    """
    读取点云文件（.pcd/.ply/.xyz/.obj 等），并返回 open3d.geometry.PointCloud 对象。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"无法读取文件 {file_path} 或点云为空。")
    return pcd


def preprocess_point_cloud(pcd, voxel_size=0.01):
    """
    对点云进行体素下采样，并估计法线。

    :param pcd: 输入的点云
    :param voxel_size: 体素大小 (单位：m)
    :return: 下采样后的点云，FPFH 特征
    """
    # 下采样
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 法线估计
    radius_normal = voxel_size * 5
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )

    # 计算 FPFH 特征
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100
        )
    )

    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,
                                distance_threshold=0.05):
    """
    基于特征匹配的全局 RANSAC 匹配，得到初始变换。

    :param source_down: 下采样后的源点云
    :param target_down: 下采样后的目标点云
    :param source_fpfh: 源点云的 FPFH 特征
    :param target_fpfh: 目标点云的 FPFH 特征
    :param distance_threshold: 匹配时用于距离检验的阈值
    :return: RANSAC 估计的配准结果 (o3d.pipelines.registration.RegistrationResult)
    """
    # 对应关系检查器
    checker = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ]

    # 使用 RANSAC 进行全局配准
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
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
    return result


def refine_registration(source, target, initial_transform, distance_threshold=0.02):
    """
    利用 ICP 算法对初始变换进行精细配准。

    :param source: 源点云（可以使用原始或较少下采样的点云）
    :param target: 目标点云（可以使用原始或较少下采样的点云）
    :param initial_transform: 初始变换矩阵 (4x4)
    :param distance_threshold: 最大对应点距离阈值
    :return: ICP 估计的精细配准结果 (o3d.pipelines.registration.RegistrationResult)
    """
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
    将 3x3 旋转矩阵转换为 Z-Y-X 欧拉角 (roll, pitch, yaw) 或其他常见顺序的欧拉角。
    这里演示使用的是“XYZ”顺序或“RPY”顺序需要自行设置，你可以根据需求调整。

    注意：在机器人学中不同软件对欧拉角的定义不尽相同，需要根据实际需求进行调整。
    这里只是一个示例。
    """
    # 以本示例为“XYZ”顺序 (roll, pitch, yaw)
    # roll  = X 轴旋转
    # pitch = Y 轴旋转
    # yaw   = Z 轴旋转

    # 若需 Z-Y-X 顺序，请在此处根据需要修改
    # 下面示例是一种常见的解法，你可能需要根据自己的定义来做调整

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])  # X 轴
        pitch = math.atan2(-R[2, 0], sy)  # Y 轴
        yaw = math.atan2(R[1, 0], R[0, 0])  # Z 轴
    else:
        # 接近奇异点
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw


def get_6d_pose_from_transformation(transformation):
    """
    从 4x4 齐次变换矩阵中提取 (x, y, z, roll, pitch, yaw)。

    :param transformation: 4x4 矩阵
    :return: (x, y, z, roll, pitch, yaw)
    """
    # 平移
    x, y, z = transformation[0, 3], transformation[1, 3], transformation[2, 3]

    # 旋转矩阵
    R = transformation[:3, :3]

    # 欧拉角
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)

    return x, y, z, roll, pitch, yaw


def main():
    # ==== 1. 读取点云 ====
    object_pcd_path = "object.pcd"  # 物体点云路径
    scene_pcd_path = "scene.pcd"  # 环境点云路径

    source_raw = load_point_cloud(object_pcd_path)
    target_raw = load_point_cloud(scene_pcd_path)

    # ==== 2. 预处理点云（下采样、法线估计、特征提取）====
    voxel_size = 0.01  # 根据实际情况进行调整
    source_down, source_fpfh = preprocess_point_cloud(source_raw, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_raw, voxel_size)

    # ==== 3. 全局 RANSAC 配准，获得初始变换 ====
    distance_threshold = voxel_size * 5.0
    print("开始全局 RANSAC 配准...")
    result_ransac = execute_global_registration(
        source_down, target_down,
        source_fpfh, target_fpfh,
        distance_threshold=distance_threshold
    )
    print("RANSAC 初始变换：\n", result_ransac.transformation)

    # ==== 4. ICP 精细配准，获取最终变换 ====
    print("开始 ICP 精细配准...")
    result_icp = refine_registration(
        source_raw, target_raw,
        result_ransac.transformation,
        distance_threshold=distance_threshold / 2.0
    )
    final_transformation = result_icp.transformation
    print("ICP 精细配准后的变换矩阵：\n", final_transformation)

    # ==== 5. 提取 6D 位姿 (x, y, z, roll, pitch, yaw) ====
    x, y, z, roll, pitch, yaw = get_6d_pose_from_transformation(final_transformation)

    print("\n========== 最终匹配结果 ==========")
    print(f"平移：x = {x:.4f}, y = {y:.4f}, z = {z:.4f}")
    print(f"旋转（欧拉角，单位：弧度）：roll = {roll:.4f}, pitch = {pitch:.4f}, yaw = {yaw:.4f}")
    print("=================================")

    # 如果需要可视化，可以取消下面注释：
    # visualize_registration(source_raw, target_raw, final_transformation)


# 如果你想要可视化，可以额外编写一个可视化函数
def visualize_registration(source, target, transformation=np.eye(4)):
    """
    可视化配准结果：将源点云变换后与目标点云一起显示。
    """
    source_temp = source.clone()
    source_temp.transform(transformation)

    source_temp.paint_uniform_color([1, 0, 0])  # 源点云红色
    target_temp = target.clone()
    target_temp.paint_uniform_color([0, 1, 0])  # 目标点云绿色

    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      width=800, height=600)


if __name__ == "__main__":
    main()
