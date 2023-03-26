import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import imageio
import sys, os, cv2

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

def pcd_gen(source_pts,source_rgb,source_normal=None):
    points = source_pts
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(source_rgb)
    if source_normal is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(source_normal)
    point_cloud.remove_duplicated_points()
    o3d.visualization.draw_geometries([point_cloud])
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.001) # voxel for mesh
    #o3d.visualization.draw_geometries([point_cloud])
    #cl, ind = point_cloud.remove_radius_outlier(nb_points=5, radius=0.01) # outlier removal
    cl, ind =point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    point_cloud = point_cloud.select_by_index(ind)
    o3d.visualization.draw_geometries([point_cloud])
    #display_inlier_outlier(voxel_down_pcd, ind)
    #o3d.visualization.draw_geometries([point_cloud])
    #cl, ind = point_cloud.remove_radius_outlier(nb_points=10, radius=0.1) # outlier removal
    #point_cloud = point_cloud.select_by_index(ind)
    #o3d.visualization.draw_geometries([point_cloud])
    #o3d.io.write_point_cloud("./test.ply", point_cloud)
    
    
    # generate mesh from pcd alpha
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
    alpha = 0.006
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        point_cloud, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    #mesh.paint_uniform_color([0.5,0.5,0.5])
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


    #compute normals
    #point_cloud.estimate_normals()
    #o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
    #o3d.io.write_point_cloud("./test.ply", point_cloud)
    #point_cloud.orient_normals_consistent_tangent_plane(100)
    #o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)

    #Poisson surface reconstruction
    #print('run Poisson surface reconstruction')
    #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #    point_cloud, depth=8)
    #o3d.visualization.draw_geometries([mesh])

    #pivoiting sruface recon
    #radii = [0.05, 0.1, 0.2, 0.4]
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #point_cloud, o3d.utility.DoubleVector(radii))
    #o3d.visualization.draw_geometries([mesh])

def test():
    data = np.load('./logs/blender/lego/lego_train/202303221805/visualization/obj_resize_0.5.npy')
    #data = np.load('./logs/blender/lego/lego_train/202303221805/visualization/lego_train_gt.npy')
    #data = np.load('./logs/replica/room_1/202303241857/visualization/room_1_gt.npy')
    #data = np.load('./logs/replica/room_1/202303241857/visualization/obj_resize_0.5.npy')
    #data = np.load('./logs/blender/hotdog/hotdog_train/202303231012/visualization/obj_resize_0.2.npy')

    pts = data[...,:3]
    z_min = np.min(pts[...,2])
    print(z_min)
    color = data[...,3:6]
    #color[:,:] = 192/255. 
    if data.shape[1]>6:
        normal = data[...,6:]
    else:
        normal = None
    pcd_gen(pts,color,normal)

#test()
