import open3d as o3d

# 1. Load sample data
sample = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(sample.path)  # <-- need .path

print("Mesh loaded!")
print(f"Vertices: {len(mesh.vertices)}")
print(f"Triangles: {len(mesh.triangles)}")

# 2. Convert to point cloud
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=5000)

# 3. Clean
pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# 4. Estimate normals
pcd_clean.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)

# 5. Reconstruct mesh
mesh_out, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=9)
mesh_out.compute_vertex_normals()

# 6. Visualize & export
o3d.visualization.draw_geometries([mesh_out])
o3d.io.write_triangle_mesh("output_model.obj", mesh_out)
print("Done! Saved output_model.obj")
