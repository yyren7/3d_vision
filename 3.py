import open3d as o3d
import numpy as np
import cv2
file_path = "situation1.asc"
points = np.loadtxt(file_path, delimiter=',', usecols=(0, 1, 2))

# 2. 创建一个 open3d 点云对象并赋值
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 3. 可视化点云
rgb_image = cv2.imread("image.bmp")

fx = 1000  # 焦距fx
fy = 1000  # 焦距fy
cx = 1296   # 图像主点的u坐标（图像宽度的一半）
cy = 972   # 图像主点的v坐标（图像高度的一半）
# 假设这里点云中的每个点已经能通过一定方式映射到图像中的像素位置
# 我们简单取整数的图像位置作为示例（具体映射需要视情况处理）
u = ((points[:, 0]+148)*2592/286 ).astype(int)
v = ((points[:, 1]+21)*1944/77 ).astype(int)

print(u,v)
# 确保(u, v)不超出图像边界
u = np.clip(u, 200, rgb_image.shape[1] - 200)
v = np.clip(v, 200, rgb_image.shape[0] - 200)

# 3. 获取对应像素的颜色并赋予点云
colors = rgb_image[1944-v, u, :] / 255.0  # 将颜色值归一化到[0, 1]

# 为点云添加颜色属性
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# 4. 显示带颜色的点云
o3d.visualization.draw_geometries([point_cloud])