import open3d as o3d
import numpy as np
# 读取 PLY 文件
# pcd = o3d.io.read_point_cloud("RGBDPoints_1733466519911547.ply")
pcd = o3d.io.read_point_cloud("../output1.ply")
# 输出点云的基本信息
print(pcd)
print("点云包含的点数:", len(pcd.points))
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)  # 设置窗口宽度和高度
vis.add_geometry(pcd)
ctr = vis.get_view_control()

ctr.set_front([0, 0, -1])  # 设置视点方向
ctr.set_lookat([0, 0, 0])  # 设置观察中心
ctr.set_up([0, -1, 0])     # 设置视点向上方向
ctr.set_zoom(1)          # 设置缩放

# 渲染窗口
vis.run()
vis.destroy_window()