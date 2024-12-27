import open3d as o3d
import numpy as np
import math
import time
from tqdm import tqdm
import copy


def load_point_cloud(file_path):
    """
    读取点云文件（包括 .ply, .pcd, .xyz, .obj 等），并返回 open3d.geometry.PointCloud 对象。
    如果点云为空或文件无效会报错。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"无法读取文件 {file_path} 或点云为空。")
    return pcd


def remove_backside_points(pcd, camera_direction=np.array([0, 0, 1]), threshold=0.0):
    """
    根据和相机视角(或某个方向向量)的点云法线/朝向关系，简单去除“背面点”。
    本例仅做一个示例：若点与指定方向的点积 < threshold，则认为是背面或不可见，可删除。

    参数:
        pcd: open3d.geometry.PointCloud，要求已经估计过法线
        camera_direction: 相机或视角的朝向向量 (默认[0, 0, 1] 表示沿 +Z 方向)
        threshold: 点积阈值，若点云法线与 camera_direction 点积 < threshold，就删除该点

    说明:
    - 这是一种简单粗略的方法，要求你的物体法线已经正确朝外，并且知道相机大致方向。
    - 如果你只想通过坐标范围 (如 z > 0) 过滤，也可以直接用 BoundingBox 等方法。
    """
    # 归一化 camera_direction
    dir_norm = camera_direction / np.linalg.norm(camera_direction)

    # 获取法线
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    keep_indices = []
    for i, n in enumerate(normals):
        dot_val = np.dot(n, dir_norm)  # n·dir_norm
        if dot_val > threshold:
            keep_indices.append(i)

    # 根据 keep_indices 生成新的点云
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points[keep_indices])

    # 若有颜色
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd_filtered.colors = o3d.utility.Vector3dVector(colors[keep_indices])

    # 法线也保留
    new_normals = normals[keep_indices]
    pcd_filtered.normals = o3d.utility.Vector3dVector(new_normals)

    return pcd_filtered


def preprocess_point_cloud(pcd, voxel_size=0.01):
    """
    对点云进行体素下采样，并估计法线，随后计算 FPFH 特征。

    :param pcd: 输入点云
    :param voxel_size: 体素下采样大小（单位：米，根据你实际场景调整）
    :return: (pcd_down, fpfh) 下采样后的点云、以及计算好的 FPFH 特征
    """
    # 下采样
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 法线估计
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )
    # 视需求可执行 normalize_normals()，也可保留默认

    # 计算 FPFH 特征
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
    checker = [
        # 几何一致性检查 (边长)
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        # 距离检查
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ]

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
    import math
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])  # X
        pitch = math.atan2(-R[2, 0], sy)  # Y
        yaw = math.atan2(R[1, 0], R[0, 0])  # Z
    else:
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


def visualize_pcd_list(geometries, window_name="Open3D"):
    """
    可视化给定的一组几何数据（点云或其他），带窗口名称。
    """
    o3d.visualization.draw_geometries(geometries,
                                      width=800,
                                      height=600,
                                      window_name=window_name)


def main():
    # === Step 1: 读取与预处理数据 ===
    with tqdm(total=100, desc="Step 1: Load Data") as pbar:
        template_ply_path = "../resources/3d/output1.ply"  # 物体(模板)点云, binary_little_endian + double
        scene_ply_path = "../resources/3d/RGBDPoints_1733466519911547.ply"  # 环境(场景)点云, ascii + float

        # 读取原始点云
        source_raw = load_point_cloud(template_ply_path)
        target_raw = load_point_cloud(scene_ply_path)

        # 小示例：估计法线，为后续“去背面”做准备
        source_raw.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.02, max_nn=30
        ))
        target_raw.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.02, max_nn=30
        ))

        for _ in range(5):
            time.sleep(0.1)
            pbar.update(20)

    # === (可选) 去掉物体背面点 ===
    # 假设场景相机朝向 +Z 方向，我们只保留 Source 中法线与 [0,0,1] 点积>0 的点
    # threshold 可根据你物体法线方向、相机姿态进行设置
    source_front = remove_backside_points(source_raw, camera_direction=np.array([0, 0, 1]), threshold=0.0)

    # === 可视化 “滤除背面后” 的 Source 与 Target ===
    source_temp = copy.deepcopy(source_front)
    source_temp.paint_uniform_color([1, 0, 0])  # 红色
    target_temp = copy.deepcopy(target_raw)
    target_temp.paint_uniform_color([0, 1, 0])  # 绿色
    visualize_pcd_list([source_temp, target_temp], window_name="Source(front-only) vs. Target")

    # === Step 2: 下采样 + 计算 FPFH + RANSAC (粗定位) ===
    with tqdm(total=100, desc="Step 2: Coarse RANSAC") as pbar:
        voxel_size = 0.01
        s_down, s_fpfh = preprocess_point_cloud(source_front, voxel_size)
        t_down, t_fpfh = preprocess_point_cloud(target_raw, voxel_size)

        distance_threshold = voxel_size * 5.0
        result_ransac = execute_global_registration(
            s_down, t_down, s_fpfh, t_fpfh,
            distance_threshold=distance_threshold
        )
        for _ in range(5):
            time.sleep(0.1)
            pbar.update(20)

    print("RANSAC 初始变换：\n", result_ransac.transformation)

    # === 可视化: RANSAC 粗配准后 ===
    source_ransac_temp = copy.deepcopy(source_front)
    source_ransac_temp.transform(result_ransac.transformation)
    source_ransac_temp.paint_uniform_color([1, 0, 0])
    target_ransac_temp = copy.deepcopy(target_raw)
    target_ransac_temp.paint_uniform_color([0, 1, 0])
    visualize_pcd_list([source_ransac_temp, target_ransac_temp], window_name="RANSAC Coarse Alignment")

    # === Step 3: 基于粗定位的 ROI 裁剪 => 只在物体附近做 ICP ===
    with tqdm(total=100, desc="Step 3: Local ICP") as pbar:

        # 1) 把物体用 RANSAC 变换到场景坐标系，然后获取其 AABB (Axis-Aligned Bounding Box)
        source_in_scene = copy.deepcopy(source_front)
        source_in_scene.transform(result_ransac.transformation)
        aabb = source_in_scene.get_axis_aligned_bounding_box()

        # 2) 在场景中截取 AABB 范围附近的点，以减少 ICP 负担
        # 可以加上一点 margin
        margin = 0.02
        aabb_min = aabb.min_bound - margin
        aabb_max = aabb.max_bound + margin
        aabb_margined = o3d.geometry.AxisAlignedBoundingBox(aabb_min, aabb_max)
        target_cropped = target_raw.crop(aabb_margined)

        # 3) ICP
        distance_threshold_icp = distance_threshold / 2.0
        result_icp = refine_registration(
            source_in_scene,  # 这时 source_in_scene 已在场景坐标系
            target_cropped,  # 仅在 AABB 范围内
            np.eye(4),  # 因为 source_in_scene 已经变换过，所以初始位姿用单位阵
            distance_threshold=distance_threshold_icp
        )

        # 4) 最终变换 = RANSAC 变换 * ICP 增量
        # RANSAC 已经把 source_front -> scene
        # ICP 是在此基础上再做微调(在 scene 坐标系下)
        # 其 result_icp.transformation 是相对于 source_in_scene 的局部变换
        final_transformation = result_icp.transformation @ result_ransac.transformation

        for _ in range(5):
            time.sleep(0.1)
            pbar.update(20)

    print("ICP 增量变换矩阵：\n", result_icp.transformation)
    print("最终变换矩阵：\n", final_transformation)

    # === 可视化最终结果 ===
    source_final_temp = copy.deepcopy(source_front)
    source_final_temp.transform(final_transformation)
    source_final_temp.paint_uniform_color([1, 0, 0])
    target_final_temp = copy.deepcopy(target_raw)
    target_final_temp.paint_uniform_color([0, 1, 0])
    visualize_pcd_list([source_final_temp, target_final_temp], window_name="Final Registration")

    # === 提取最终 6D 位姿并打印 ===
    x, y, z, roll, pitch, yaw = get_6d_pose_from_transformation(final_transformation)
    print("\n========== 最终匹配结果 ==========")
    print(f"平移：x = {x:.4f}, y = {y:.4f}, z = {z:.4f}")
    print(f"旋转（欧拉角，单位：弧度）：roll = {roll:.4f}, pitch = {pitch:.4f}, yaw = {yaw:.4f}")
    print("=================================")


if __name__ == "__main__":
    main()
