#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云查看器 - Open3D交互式查看器
用法: python view_pointcloud.py [点云文件路径.ply/pcd/obj]

支持的格式:
- PLY (.ply)
- PCD (.pcd)
- OBJ (.obj)

交互控制:
- 鼠标左键: 旋转
- 鼠标右键/Ctrl+左键: 平移
- 鼠标滚轮: 缩放
- '['/']': 减小/增大点大小
- 'R': 重置视图
- 'B': 切换背景颜色
- 'S': 保存截图
- 'I': 显示点云信息
- 'Q'或ESC: 退出
"""

import os
import sys
import numpy as np
import open3d as o3d
import glob

def load_point_cloud(file_path):
    """
    加载点云文件
    
    参数:
        file_path: 点云文件路径 (.ply, .pcd, .obj)
    
    返回:
        point_cloud: Open3D点云对象
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.ply':
        return o3d.io.read_point_cloud(file_path)
    elif file_extension == '.pcd':
        return o3d.io.read_point_cloud(file_path)
    elif file_extension == '.obj':
        mesh = o3d.io.read_triangle_mesh(file_path)
        # 从网格中提取点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        else:
            # 如果网格没有颜色，使用默认颜色
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
        return pcd
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")

def print_cloud_info(point_cloud):
    """
    打印点云信息
    
    参数:
        point_cloud: Open3D点云对象
    """
    print("========== 点云信息 ==========")
    print(f"点数量: {len(point_cloud.points)}")
    print(f"点密度: {len(point_cloud.points) / 1000:.2f}k点")
    
    # 检查是否有颜色
    if point_cloud.has_colors():
        print("包含颜色: 是")
    else:
        print("包含颜色: 否")
    
    # 检查是否有法线
    if point_cloud.has_normals():
        print("包含法线: 是")
    else:
        print("包含法线: 否")
    
    # 计算包围盒
    min_bound = point_cloud.get_min_bound()
    max_bound = point_cloud.get_max_bound()
    dimensions = max_bound - min_bound
    center = point_cloud.get_center()
    
    print("\n包围盒信息:")
    print(f"  最小点: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}]")
    print(f"  最大点: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}]")
    print(f"  尺寸: [{dimensions[0]:.3f}, {dimensions[1]:.3f}, {dimensions[2]:.3f}]")
    print(f"  中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print("==============================")

def visualize_point_cloud(point_cloud, window_name="点云查看器"):
    """
    交互式可视化点云
    
    参数:
        point_cloud: Open3D点云对象
        window_name: 窗口标题
    """
    # 如果点云没有颜色，添加默认颜色
    if not point_cloud.has_colors():
        point_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    
    # 初始化可视化器
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=1024, height=768)
    vis.add_geometry(point_cloud)
    
    # 渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 1.0
    
    # 当前点尺寸
    point_size = 1.0
    # 背景颜色索引 (0: 黑色, 1: 白色, 2: 灰色)
    bg_color_idx = 0
    bg_colors = [
        np.array([0.1, 0.1, 0.1]),  # 深灰色/黑色
        np.array([1.0, 1.0, 1.0]),  # 白色
        np.array([0.5, 0.5, 0.5]),  # 中灰色
    ]
    
    # 信息显示函数
    def show_info():
        print_cloud_info(point_cloud)
    
    # 注册键盘回调
    def change_point_size_smaller(vis):
        nonlocal point_size
        point_size = max(0.1, point_size - 0.1)
        opt.point_size = point_size
        print(f"点大小: {point_size:.1f}")
        return True
    
    def change_point_size_larger(vis):
        nonlocal point_size
        point_size += 0.1
        opt.point_size = point_size
        print(f"点大小: {point_size:.1f}")
        return True
    
    def reset_view(vis):
        vis.reset_view_point(True)
        print("视图已重置")
        return True
    
    def change_background(vis):
        nonlocal bg_color_idx
        bg_color_idx = (bg_color_idx + 1) % len(bg_colors)
        opt.background_color = bg_colors[bg_color_idx]
        print(f"背景颜色已更改")
        return True
    
    def save_screenshot(vis):
        timestamp = int(np.floor(np.datetime64('now').astype(np.int64) / 1e6))
        image_path = f"screenshot_{timestamp}.png"
        vis.capture_screen_image(image_path, True)
        print(f"截图已保存: {image_path}")
        return True
    
    def display_info(vis):
        show_info()
        return True
    
    def exit_callback(vis):
        print("退出查看器")
        vis.destroy_window()
        return False
    
    # 注册键盘回调函数
    vis.register_key_callback(ord('['), change_point_size_smaller)
    vis.register_key_callback(ord(']'), change_point_size_larger)
    vis.register_key_callback(ord('R'), reset_view)
    vis.register_key_callback(ord('B'), change_background)
    vis.register_key_callback(ord('S'), save_screenshot)
    vis.register_key_callback(ord('I'), display_info)
    vis.register_key_callback(ord('Q'), exit_callback)
    vis.register_key_callback(27, exit_callback)  # ESC键
    
    # 初始信息显示
    show_info()
    print("交互式控制:")
    print("- 鼠标左键: 旋转")
    print("- 鼠标右键/Ctrl+左键: 平移")
    print("- 鼠标滚轮: 缩放")
    print("- '['/']': 减小/增大点大小")
    print("- 'R': 重置视图")
    print("- 'B': 切换背景颜色")
    print("- 'S': 保存截图")
    print("- 'I': 显示点云信息")
    print("- 'Q'或ESC: 退出")
    
    # 运行可视化循环
    vis.run()
    vis.destroy_window()

def find_3d_files(directory="."):
    """
    在指定目录查找所有支持的3D文件
    
    参数:
        directory: 要搜索的目录
    
    返回:
        支持的3D文件列表
    """
    supported_extensions = ['.ply', '.pcd', '.obj']
    all_files = []
    
    for ext in supported_extensions:
        pattern = os.path.join(directory, f"*{ext}")
        all_files.extend(glob.glob(pattern))
        
        # 检查子目录 image
        image_dir = os.path.join(directory, "image")
        if os.path.exists(image_dir) and os.path.isdir(image_dir):
            pattern = os.path.join(image_dir, f"*{ext}")
            all_files.extend(glob.glob(pattern))
    
    # 按字母顺序排序
    all_files.sort()
    return all_files

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        # 如果没有提供文件路径，显示image目录中的3D文件
        print(f"正在搜索3D文件...")
        
        # 直接查找image目录中的文件
        image_dir = os.path.join(os.getcwd(), "image")
        if not os.path.exists(image_dir):
            print(f"错误: 找不到image目录: {image_dir}")
            return
            
        # 查找所有PLY文件
        ply_files = glob.glob(os.path.join(image_dir, "*.ply"))
        if not ply_files:
            print("错误: image目录中未找到PLY文件")
            return
            
        # 按名称排序
        ply_files.sort()
        
        # 显示文件列表
        print("\n找到以下3D文件:")
        for i, file_path in enumerate(ply_files):
            # 显示文件名而不是完整路径
            file_name = os.path.basename(file_path)
            print(f"[{i+1}] {file_name}")
        
        # 让用户选择文件
        while True:
            try:
                choice = input("\n请输入文件编号: ")
                if not choice.strip():  # 检查是否为空输入
                    print("请输入有效的文件编号")
                    continue
                    
                choice_num = int(choice)
                if choice_num < 1 or choice_num > len(ply_files):
                    print(f"错误: 无效的选择。请输入1到{len(ply_files)}之间的数字。")
                    continue
                
                file_path = ply_files[choice_num-1]
                break  # 成功选择后退出循环
            except ValueError:
                print("错误: 请输入有效的数字。")
    else:
        file_path = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return
    
    try:
        # 加载点云
        print(f"正在加载点云: {file_path}")
        point_cloud = load_point_cloud(file_path)
        
        # 如果点云为空，则退出
        if len(point_cloud.points) == 0:
            print("错误: 点云为空")
            return
        
        # 可视化点云
        visualize_point_cloud(point_cloud, f"点云查看器 - {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return

if __name__ == "__main__":
    main() 