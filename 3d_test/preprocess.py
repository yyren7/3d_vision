import copy
import open3d as o3d
import numpy as np


def interactive_crop_and_save(input_path, output_path):
    """
    交互式点云截取并保存工具
    参数：
        input_path: 输入点云文件路径（支持ply/pcd等格式）
        output_path: 输出点云保存路径（推荐ply格式）
    """
    # 1. 加载点云数据
    pcd = o3d.io.read_point_cloud(input_path)
    if pcd.is_empty():
        raise ValueError("无法加载点云文件，请检查路径和文件格式")

    print(f"原始点云信息：")
    print(f"- 点数：{len(pcd.points)}")
    print(f"- 包围盒尺寸：{pcd.get_axis_aligned_bounding_box().get_extent()}")

    # 2. 交互式点云选择
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="tool", width=1920, height=1080)
    vis.add_geometry(pcd)
    
    print("\n操作指南：")
    print("1. 按 K 键进入选择模式")
    print("2. 按住 Shift + 鼠标左键绘制选区多边形")
    print("3. 按 C 键确认选择区域")
    print("4. 按 Q 键退出并保存选区")
    
    vis.run()  # 进入交互模式
    
    # 3. 获取选区索引
    picked_indices = vis.get_picked_points()
    if not picked_indices:
        raise RuntimeError("未选择任何区域，请重新操作")

    # 4. 提取选区点云
    selected_pcd = pcd.select_by_index(picked_indices)
    
    # 5. 后处理流程
    def post_process(cloud):
        # 去噪处理
        cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # 降采样
        downsampled = cl.voxel_down_sample(voxel_size=0.005)
        # 法线估计
        downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        return downsampled

    processed_pcd = post_process(selected_pcd)
    
    print("\n选区信息：")
    print(f"- 截取点数：{len(selected_pcd.points)}")
    print(f"- 处理后点数：{len(processed_pcd.points)}")
    print(f"- 新包围盒尺寸：{processed_pcd.get_axis_aligned_bounding_box().get_extent()}")

    # 6. 可视化验证
    original_pcd = copy.deepcopy(pcd).paint_uniform_color([0.8, 0.8, 0.8])  # 灰色原始点云
    processed_pcd.paint_uniform_color([1, 0, 0])  # 红色显示选区
    o3d.visualization.draw_geometries([original_pcd, processed_pcd], 
                                    window_name="选区验证")

    # 7. 保存结果
    o3d.io.write_point_cloud(output_path, processed_pcd)
    print(f"\n成功保存截取点云至：{output_path}")

if __name__ == "__main__":
    # 使用示例 - 修改为您的实际路径
    input_file = "/home/ncpt-am/local_model/resources/3d/scene.ply"       # 原始场景点云路径
    output_file = "/home/ncpt-am/local_model/resources/3d/scene_cropped.ply"    # 截取结果保存路径
    
    interactive_crop_and_save(input_file, output_file)