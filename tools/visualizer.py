import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

points = np.random.rand(10000, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(np.random.rand(10000, 3))
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
alpha = 1.5
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    point_cloud, alpha, tetra_mesh, pt_map)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

#o3d.visualization.draw_geometries([point_cloud])

#ply = o3d.geometry.TriangleMesh()
#ply.vertices = o3d.utility.Vector3dVector(points)
#ply.triangles = o3d.utility.Vector3iVector(tri)
#o3d.visualization.draw_geometries([ply])