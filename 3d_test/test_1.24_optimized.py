# %% [markdown]
# ## 点云粗配准流程 - 鲁棒优化版
# 本 Notebook 实现基于RANSAC和ICP的点云配准，增加了多次尝试和结果验证机制

# %% [markdown]
# ### 1. 导入依赖库

# %%
import open3d as o3d
import numpy as np
import copy
import time
import random
import os
from scipy.spatial.transform import Rotation
from datetime import datetime

# 设置随机种子以确保结果可重复
np.random.seed(42)
random.seed(42)

# 创建image文件夹（如果不存在）
image_dir = "/home/ncpt-am/local_model/image"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"创建图像保存目录: {image_dir}")
else:
    print(f"图像将保存至: {image_dir}")

# %% [markdown]
# ### 2. 定义可视化函数

# %%
def custom_draw_geometry(pcd_list, window_name="Open3D", wait_time=2.0, save_image=False, image_filename=None):
    """简单的可视化函数，自动显示几秒后关闭窗口，并可选保存图像"""
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    
    # 添加点云
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    
    # 设置视图
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 暗灰色背景
    opt.point_size = 2.0  # 增大点的尺寸
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # 更新渲染
    vis.poll_events()
    vis.update_renderer()
    
    # 保存图像（如果需要）
    if save_image and image_filename:
        image_path = os.path.join(image_dir, image_filename)
        vis.capture_screen_image(image_path, True)
        print(f"图像已保存: {image_path}")
    
    # 显示指定时间
    print(f"显示窗口：{window_name}，{wait_time}秒后自动关闭...")
    start_time = time.time()
    while time.time() - start_time < wait_time:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)
    
    # 关闭窗口
    vis.destroy_window()

# %% [markdown]
# ### 3. 数据加载与初始化

# %%
# 加载点云
print("正在加载点云...")
object_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/cropped_1.ply")
scene_raw = o3d.io.read_point_cloud("/home/ncpt-am/local_model/resources/3d/scene_cropped_1.ply")

# 对object_raw进行翻转（沿Z轴翻转180度）
print("翻转物体点云...")
R = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]  # Z轴翻转
])
object_raw.rotate(R, center=object_raw.get_center())

# 点云尺度标准化
scale_factor = 0.01  # 假设原始数据单位为厘米，转换为米
object_raw.scale(scale_factor, center=(0,0,0))
scene_raw.scale(scale_factor, center=(0,0,0))

# 初始位置调整 - 使用更保守的平移
translation = scene_raw.get_center() - object_raw.get_center()
object_raw.translate(translation * 0.5)

# 颜色标记
object_colored = copy.deepcopy(object_raw).paint_uniform_color([1, 0, 0])
scene_colored = copy.deepcopy(scene_raw).paint_uniform_color([0, 0, 1])

print("\n初始位置可视化：")
custom_draw_geometry([object_colored, scene_colored])

# %% [markdown]
# ### 4. 数据预处理 - 增强版

# %%
# 降采样与离群点去除函数
def preprocess_point_cloud(pcd, voxel_size):
    """增强的点云预处理函数"""
    print(f"[预处理] 使用体素大小 {voxel_size}m 降采样...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    print("[预处理] 第一轮离群点去除...")
    pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
    
    print("[预处理] 第二轮离群点去除...")
    pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    
    print("[预处理] 高精度法线估计...")
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size*3, max_nn=50))
    
    # 确保法线方向一致性
    pcd_down.orient_normals_consistent_tangent_plane(30)
    
    return pcd_down

# 使用优化的体素大小
voxel_size = 0.02
object_down = preprocess_point_cloud(object_raw, voxel_size)
scene_down = preprocess_point_cloud(scene_raw, voxel_size)

print("\n预处理可视化：")
object_down_vis = copy.deepcopy(object_down).paint_uniform_color([1, 0, 0])
scene_down_vis = copy.deepcopy(scene_down).paint_uniform_color([0, 0, 1])
custom_draw_geometry([object_down_vis, scene_down_vis])

# %% [markdown]
# ### 5. 特征提取与鲁棒配准

# %%
# 优化的FPFH特征提取
def compute_enhanced_fpfh(pcd, voxel_size):
    """增强的FPFH特征计算"""
    radius_feature = voxel_size * 8  # 适度调整特征半径
    print(f"[特征提取] 计算FPFH特征（半径={radius_feature:.3f}m）...")
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=150  # 减少最大邻居点数以平衡计算效率和特征质量
        )
    )

# 多次RANSAC尝试，选择最佳结果
def robust_ransac_registration(source, target, source_fpfh, target_fpfh, voxel_size, num_attempts=5):
    """执行多次RANSAC，选择最佳结果"""
    best_result = None
    best_fitness = 0
    
    print(f"\n[粗配准] 执行多次RANSAC尝试 (尝试次数={num_attempts})...")
    
    # 定义不同的RANSAC参数组合
    ransac_params = [
        # max_correspondence_distance, ransac_n, max_iteration
        (voxel_size * 6, 4, 1000000),  # 默认参数
        (voxel_size * 4, 3, 800000),   # 更严格的对应距离
        (voxel_size * 8, 5, 1200000),  # 更宽松的对应距离
        (voxel_size * 5, 4, 1500000),  # 更多迭代
        (voxel_size * 7, 3, 500000),   # 不同组合
    ]
    
    for attempt in range(num_attempts):
        # 使用不同参数组合
        param_idx = attempt % len(ransac_params)
        max_dist, ransac_n, max_iter = ransac_params[param_idx]
        
        print(f"  - 尝试 {attempt+1}/{num_attempts} (max_dist={max_dist:.3f}, ransac_n={ransac_n}, max_iter={max_iter})")
        
        # 执行RANSAC
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target,
            source_fpfh, target_fpfh,
            mutual_filter=True,  # 启用互斥滤波提高匹配质量
            max_correspondence_distance=max_dist,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=ransac_n,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999)
        )
        
        # 判断是否是迄今为止最佳结果
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_result = result
            print(f"    * 新的最佳得分: {best_fitness:.3f}")
    
    # 检查结果合理性 - 避免极端变换
    if best_result is not None:
        # 提取旋转部分 - 创建副本而不是视图，解决只读数组问题
        rotation_matrix = np.array(best_result.transformation[:3, :3], copy=True)
        translation = np.array(best_result.transformation[:3, 3], copy=True)
        
        # 计算旋转角度
        r = Rotation.from_matrix(rotation_matrix)
        angles = np.abs(r.as_euler('xyz', degrees=True))
        max_angle = np.max(angles)
        
        # 计算平移距离
        translation_distance = np.linalg.norm(translation)
        
        print(f"\n[验证] 最大旋转角度: {max_angle:.2f}°, 平移距离: {translation_distance:.2f}m")
        
        # 如果变换太极端，可能是错误结果
        if max_angle > 170 or translation_distance > 20:
            print("警告: 变换参数异常，可能是错误的配准结果!")
    
    return best_result

# 计算特征
object_fpfh = compute_enhanced_fpfh(object_down, voxel_size)
scene_fpfh = compute_enhanced_fpfh(scene_down, voxel_size)

# 执行多次RANSAC尝试
ransac_result = robust_ransac_registration(
    object_down, scene_down,
    object_fpfh, scene_fpfh,
    voxel_size,
    num_attempts=5  # 尝试5次，选择最佳结果
)

# 应用变换矩阵
object_ransac = copy.deepcopy(object_down).transform(ransac_result.transformation)
object_ransac.paint_uniform_color([1, 0, 0])

print("\n粗配准结果可视化：")
custom_draw_geometry([object_ransac, scene_down])

# %% [markdown]
# ### 6. 改进的ICP精配准

# %%
def robust_icp_registration(source, target, ransac_transform, voxel_size):
    """增强的多阶段ICP配准，具有回退机制"""
    print("\n[精配准] 执行鲁棒多阶段ICP...")
    
    # 存储中间结果和评估指标
    icp_results = []
    best_result = None
    best_error = float('inf')
    
    # 应用初始变换
    current_transform = ransac_transform
    current_source = copy.deepcopy(source).transform(current_transform)
    
    # 定义修订后的ICP阶段参数 - 更加稳健
    icp_stages = [
        # 阶段名称, 距离阈值, 最大迭代, 相对适应度, 相对RMSE, 点到平面, 是否回退
        ("全局粗对齐", 8.0, 100, 1e-6, 1e-6, False, False),
        ("中等精度对齐", 4.0, 80, 1e-7, 1e-7, True, True),
        ("几何细化", 2.0, 60, 1e-8, 1e-8, True, True),
        ("局部精调", 1.0, 40, 1e-9, 1e-9, True, True),
        ("最终优化", 0.5, 30, 1e-10, 1e-10, False, False)
    ]

    # 执行多阶段ICP
    for stage_idx, (stage_name, dist_threshold, max_iter, rel_fit, rel_rmse, use_plane, allow_rollback) in enumerate(icp_stages):
        print(f"\n执行阶段 {stage_idx+1}: {stage_name}...")
        
        # 重新计算法线（如果需要的话）
        if use_plane:
            current_source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size*(3-stage_idx*0.5 if stage_idx < 3 else 1), 
                    max_nn=50
                )
            )
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size*(3-stage_idx*0.5 if stage_idx < 3 else 1), 
                    max_nn=50
                )
            )
        
        # 选择估计方法
        if use_plane:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        
        # 执行ICP
        result = o3d.pipelines.registration.registration_icp(
            current_source, target, 
            voxel_size * dist_threshold,
            np.identity(4),  # 使用单位矩阵，因为我们使用的是当前已变换的点云
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter, 
                relative_fitness=rel_fit, 
                relative_rmse=rel_rmse
            )
        )
        
        # 计算当前误差
        distances = np.asarray(target.compute_point_cloud_distance(current_source.transform(result.transformation)))
        max_dist = np.max(distances)
        avg_dist = np.mean(distances)
        
        print(f"  - 得分: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.5f}")
        print(f"  - 最大误差: {max_dist:.5f}m, 平均误差: {avg_dist:.5f}m")
        
        # 检查结果是否有效 - 避免退化
        result_is_valid = True
        if result.fitness < 0.001 and stage_idx > 0:  # 忽略第一阶段的低得分
            print("  ! 警告: 配准得分过低，可能是无效结果")
            result_is_valid = False
            
        if max_dist > 20:  # 避免极端大的误差
            print("  ! 警告: 最大误差过大，可能是无效结果")
            result_is_valid = False
        
        # 回退判断 - 如果结果变差且允许回退
        if allow_rollback and avg_dist > best_error * 1.5:  # 如果误差增加50%以上
            print(f"  ! 警告: 当前误差 ({avg_dist:.5f}m) 远大于最佳误差 ({best_error:.5f}m)，执行回退")
            # 跳过此阶段，不更新变换矩阵
            result_is_valid = False
        
        # 如果结果有效，更新当前变换
        if result_is_valid:
            # 更新当前变换和点云
            current_transform = np.matmul(result.transformation, current_transform)
            current_source = copy.deepcopy(source).transform(current_transform)
            
            # 更新最佳误差
            if avg_dist < best_error:
                best_error = avg_dist
                best_result = result
            
            # 保存结果
            icp_results.append((stage_name, result, max_dist, avg_dist))
        else:
            print("  * 跳过此阶段，保持之前的变换")
    
    # 返回最佳结果
    return current_transform, icp_results

# 执行多阶段ICP
final_transform, icp_results = robust_icp_registration(
    object_down, scene_down, 
    ransac_result.transformation, 
    voxel_size
)

# 应用最终的变换矩阵
object_icp = copy.deepcopy(object_down).transform(final_transform)
object_icp.paint_uniform_color([0, 1, 0])  # 绿色表示精配准

print("\n精配准结果可视化：")
custom_draw_geometry([object_icp, scene_down], wait_time=4.0)

# %% [markdown]
# ### 7. 结果分析与可视化

# %%
print("\n=== 粗配准结果 ===")
print(f"配准得分: {ransac_result.fitness:.3f}")
print(f"变换矩阵:\n{ransac_result.transformation}")

print("\n=== 最终精配准结果 ===")
for i, (stage_name, result, max_dist, avg_dist) in enumerate(icp_results):
    print(f"阶段 {i+1} ({stage_name}): 得分={result.fitness:.3f}, RMSE={result.inlier_rmse:.5f}")
    print(f"  - 最大误差: {max_dist:.5f}m, 平均误差: {avg_dist:.5f}m")

print(f"\n最终变换矩阵:\n{final_transform}")

# 计算最终误差
final_distances = np.asarray(scene_down.compute_point_cloud_distance(object_icp))
final_max_dist = np.max(final_distances)
final_avg_dist = np.mean(final_distances)
print(f"\n最终最大误差: {final_max_dist:.5f}m")
print(f"最终平均误差: {final_avg_dist:.5f}m")

# 显示粗配准和精配准的对比
print("\n粗配准与精配准对比：")
object_ransac = copy.deepcopy(object_down).transform(ransac_result.transformation)
object_ransac.paint_uniform_color([1, 0, 0])  # 红色表示粗配准
object_icp.paint_uniform_color([0, 1, 0])     # 绿色表示精配准

# 创建叠加视图，提高对比的可视性
scene_down_vis = copy.deepcopy(scene_down).paint_uniform_color([0, 0, 1])  # 蓝色表示场景

# 生成带有时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
comparison_file_name = f"registration_comparison_{timestamp}.ply"
final_result_file_name = f"final_registration_{timestamp}.ply"
transform_file_name = f"transform_matrix_{timestamp}.txt"

# 保存对比视图的点云（将粗配准和精配准结果合并为一个点云）
comparison_cloud = o3d.geometry.PointCloud()
comparison_cloud.points = o3d.utility.Vector3dVector(
    np.vstack([
        np.asarray(object_ransac.points), 
        np.asarray(object_icp.points), 
        np.asarray(scene_down_vis.points)
    ])
)
comparison_cloud.colors = o3d.utility.Vector3dVector(
    np.vstack([
        np.asarray(object_ransac.colors),
        np.asarray(object_icp.colors),
        np.asarray(scene_down_vis.colors)
    ])
)
comparison_path = os.path.join(image_dir, comparison_file_name)
o3d.io.write_point_cloud(comparison_path, comparison_cloud)
print(f"已保存对比点云: {comparison_path}")

# 单独保存最终配准结果
final_result_cloud = o3d.geometry.PointCloud()
final_result_cloud.points = o3d.utility.Vector3dVector(
    np.vstack([
        np.asarray(object_icp.points), 
        np.asarray(scene_down_vis.points)
    ])
)
final_result_cloud.colors = o3d.utility.Vector3dVector(
    np.vstack([
        np.asarray(object_icp.colors),
        np.asarray(scene_down_vis.colors)
    ])
)
final_path = os.path.join(image_dir, final_result_file_name)
o3d.io.write_point_cloud(final_path, final_result_cloud)
print(f"已保存最终配准点云: {final_path}")

# 保存变换矩阵
transform_path = os.path.join(image_dir, transform_file_name)
np.savetxt(transform_path, final_transform, delimiter=',', header='# 最终变换矩阵')
print(f"已保存变换矩阵: {transform_path}")

# 同时显示3D可视化
custom_draw_geometry([object_ransac, object_icp, scene_down_vis], "RANSAC vs ICP对比", wait_time=5.0)
custom_draw_geometry([object_icp, scene_down_vis], "最终配准结果", wait_time=3.0)

# 输出总体评估结果
if final_avg_dist < 1.0:
    print("\n[成功] 配准完成！平均误差低于1米。")
    print(f"结果3D点云已保存至: \n- {comparison_path}\n- {final_path}")
else:
    print("\n[警告] 配准结果可能不够理想，平均误差大于1米。")
    print(f"结果3D点云已保存至: \n- {comparison_path}\n- {final_path}")
    
if final_max_dist > 3.0:
    print("[警告] 最大误差仍然较大 (>3米)，可能需要进一步检查点云质量或增加更多约束。") 