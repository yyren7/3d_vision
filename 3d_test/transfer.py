import open3d as o3d
import numpy as np

# 读取带颜色的 ASC 文件
def load_asc_with_color(filename):
    points = []
    colors = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            if len(data) >= 6:  # 确保是 6 维数据
                try:
                    x, y, z = float(data[0]), float(data[1]), float(data[2])
                    r, g, b = float(data[3]), float(data[4]), float(data[5])
                    points.append([x, y, z])
                    colors.append([r, g, b])
                except ValueError:
                    print(f"跳过无效行: {line.strip()}")
    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.float64)

# 加载 ASC 文件
points, colors = load_asc_with_color("../output1.asc")

# 检查数据
print("点云坐标形状:", points.shape)
print("颜色数据形状:", colors.shape)

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)   # 设置坐标
pcd.colors = o3d.utility.Vector3dVector(colors)   # 设置颜色 (0-1 范围)

# 显示带颜色的点云
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("../output1.ply", pcd)   # 保存为 PLY 文件