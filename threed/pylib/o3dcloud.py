import open3d as o3d
import numpy as np
import trimesh




print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("/home/samir/Open3D-master/examples/test_data/pointcl-depth.ply")
# print(pcd)
# print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])
# tooth = vol.crop_point_cloud(pcd)
# o3d.visualization.draw_geometries([tooth])
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist
pcd.estimate_normals()
# poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))

tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))

o3d.visualization.draw_geometries([mesh])
o3d.visualization.draw_geometries([tri_mesh])


print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd])

print("Let's define some primitives")
mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
                                                height=1.0,
                                                depth=1.0)



print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd])

print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd],
                                  point_show_normal=True)


# mesh_box.compute_vertex_normals()
# mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
# mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
# mesh_sphere.compute_vertex_normals()
# mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
# mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
#                                                             height=4.0)
# mesh_cylinder.compute_vertex_normals()
# mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.6, origin=[-2, -2, -2])

# print("We draw a few primitives using collection.")
# o3d.visualization.draw_geometries(
#         [mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

# print("We draw a few primitives using + operator of mesh.")
# o3d.visualization.draw_geometries(
#         [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])                                  