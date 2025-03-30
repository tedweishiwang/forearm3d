import json
import open3d as o3d
import numpy as np

# Load RealSense config
with open("config.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

# Initialize RealSense sensor
rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0, "test.bag")
rs.start_capture(True)

# Setup non-blocking visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Define camera intrinsics and extrinsics (and flip transform for correct orientation)
intrinsics = o3d.core.Tensor([
    [615.0, 0.0, 320.0],
    [0.0, 615.0, 240.0],
    [0.0, 0.0, 1.0]
], dtype=o3d.core.Dtype.Float32)
extrinsics = o3d.core.Tensor(np.eye(4), dtype=o3d.core.Dtype.Float32)
flip_transform = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

# Capture an initial frame to create the point cloud and add it to the visualizer
im_rgbd = rs.capture_frame(True, True)
pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsics, extrinsics)
pcd_legacy = pcd.to_legacy()
pcd_legacy.transform(flip_transform)
vis.add_geometry(pcd_legacy)

# Real-time update loop (runs indefinitely; press the window's close button or interrupt the process to stop)
while True:
    # Capture new frame and create a new point cloud
    im_rgbd = rs.capture_frame(True, True)
    pcd_new = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsics, extrinsics)
    pcd_new_legacy = pcd_new.to_legacy()
    pcd_new_legacy.transform(flip_transform)

    # Update the existing geometry with new data
    pcd_legacy.points = pcd_new_legacy.points
    pcd_legacy.colors = pcd_new_legacy.colors
    if pcd_new_legacy.has_normals():
        pcd_legacy.normals = pcd_new_legacy.normals

    # Convert points and colors to NumPy arrays for processing
    points = np.asarray(pcd_legacy.points)
    colors = np.asarray(pcd_legacy.colors)

    # Compute the Euclidean distance (depth) from the sensor (assumed at origin) for each point
    dists = np.linalg.norm(points, axis=1)
    # Determine threshold based on the 10th percentile (i.e. closest 10% of points)
    threshold = np.percentile(dists, 10)
    # Create mask for points with depth smaller than or equal to the threshold
    mask = dists <= threshold

    # Set the color of those points to red [R, G, B] = [1, 0, 0]
    colors[mask] = [1.0, 0.0, 0.0]
    # Update the colors of the point cloud
    pcd_legacy.colors = o3d.utility.Vector3dVector(colors)

    # Optional: Print the point cloud info for debugging
    print(pcd_legacy)

    # Update the visualizer window
    vis.update_geometry(pcd_legacy)
    vis.poll_events()
    vis.update_renderer()

# Stop sensor capture and close the window (this part is reached if the loop ends)
rs.stop_capture()
vis.destroy_window()
