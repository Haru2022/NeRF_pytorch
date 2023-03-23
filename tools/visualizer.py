import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

#points = np.random.rand(10000, 3)
#point_cloud = o3d.geometry.PointCloud()
#point_cloud.points = o3d.utility.Vector3dVector(points)
#point_cloud.colors = o3d.utility.Vector3dVector(np.random.rand(10000, 3))
#tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
#alpha = 1.5
#print(f"alpha={alpha:.3f}")
#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#    point_cloud, alpha, tetra_mesh, pt_map)
#mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

#o3d.visualization.draw_geometries([point_cloud])

#ply = o3d.geometry.TriangleMesh()
#ply.vertices = o3d.utility.Vector3dVector(points)
#ply.triangles = o3d.utility.Vector3iVector(tri)
#o3d.visualization.draw_geometries([ply])

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud])

def pcd_gen(source_pts,source_rgb):
    points = source_pts
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(source_rgb)
    cl, ind = point_cloud.remove_radius_outlier(nb_points=5, radius=0.01) # outlier removal
    #cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)
    o3d.visualization.draw_geometries([point_cloud])
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01) # voxel for mesh
    #display_inlier_outlier(voxel_down_pcd, ind)
    o3d.visualization.draw_geometries([point_cloud])
    
    # generate mesh from pcd
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
    alpha = 0.01
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        point_cloud, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


#data = np.load('./logs/blender/lego/lego_train/202303221805/visualization/obj_resize_0.5.npy')
#pts = data[...,:3]
#color = data[...,3:]
#pcd_gen(pts,color)