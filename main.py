# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import open3d as o3d
import time
import open3d.cpu.pybind.t.pipelines.registration as treg


def draw_registration_result(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp.to_legacy(),
         target_temp.to_legacy()],
        zoom=0.5,
        front=[-0.2458, -0.8088, 0.5342],
        lookat=[1.7745, 2.2305, 0.9787],
        up=[0.3109, -0.5878, -0.7468])


if __name__ == '__main__':
    source = o3d.t.io.read_point_cloud("frag_115.ply")
    target = o3d.t.io.read_point_cloud("frag_116.ply")
    init_source_to_target = np.identity(4)
    # For Colored-ICP `colors` attribute must be of the same dtype as `positions` and `normals` attribute.
    source.point["colors"] = source.point["colors"].to(
        o3d.core.Dtype.Float32) / 255.0
    target.point["colors"] = target.point["colors"].to(
        o3d.core.Dtype.Float32) / 255.0
    max_correspondence_distance = 0.02
    estimation = treg.TransformationEstimationForColoredICP()
    current_transformation = np.identity(4)
    reg_point_to_plane = treg.icp(source, target, max_correspondence_distance,
                                  init_source_to_target, estimation)
    criteria_list = [
        treg.ICPConvergenceCriteria(relative_fitness=0.0001,
                                    relative_rmse=0.0001,
                                    max_iteration=50),
        treg.ICPConvergenceCriteria(0.00001, 0.00001, 30),
        treg.ICPConvergenceCriteria(0.000001, 0.000001, 14)
    ]

    max_correspondence_distances = o3d.utility.DoubleVector([0.08, 0.04, 0.02])

    voxel_sizes = o3d.utility.DoubleVector([0.04, 0.02, 0.01])
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017

    print("Colored point cloud registration")
    s = time.time()

    reg_multiscale_icp = treg.multi_scale_icp(source, target, voxel_sizes,
                                              criteria_list,
                                              max_correspondence_distances,
                                              init_source_to_target, estimation)

    icp_time = time.time() - s
    print("Time taken by Colored ICP: ", icp_time)
    print("Fitness: ", reg_multiscale_icp.fitness)
    print("Inlier RMSE: ", reg_multiscale_icp.inlier_rmse)

    draw_registration_result(source, target, reg_multiscale_icp.transformation)
