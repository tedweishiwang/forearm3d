import json
import threading
import open3d as o3d
import numpy as np

# Create an event flag for saving
save_trigger = threading.Event()
file_num = 0

# Function to listen for Enter key press in a separate thread
def wait_for_enter():
    while True:
        input("Press Enter to capture and save the red point cloud...\n")
        save_trigger.set()

# Start the listener thread (daemon so it exits when main program ends)
save_thread = threading.Thread(target=wait_for_enter, daemon=True)
save_thread.start()

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

# Capture an initial frame to create a point cloud; this initial cloud is not used for display
im_rgbd = rs.capture_frame(True, True)
pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsics, extrinsics)
pcd_legacy = pcd.to_legacy()
pcd_legacy.transform(flip_transform)

# Add an empty geometry to the visualizer (we will update it each frame)
vis.add_geometry(pcd_legacy)

# Real-time update loop (runs indefinitely; press the window's close button or interrupt the process to stop)
while True:
    # Capture new frame and create a new point cloud
    im_rgbd = rs.capture_frame(True, True)
    pcd_new = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsics, extrinsics)
    pcd_new_legacy = pcd_new.to_legacy()
    pcd_new_legacy.transform(flip_transform)

    # Convert points to a NumPy array for filtering
    points = np.asarray(pcd_new_legacy.points)
    # Compute the Euclidean distance (depth) from the sensor for each point
    dists = np.linalg.norm(points, axis=1)
    # Determine threshold based on the 20th percentile (i.e. closest 20% of points)
    threshold = np.percentile(dists, 20)
    # Create mask for points with depth smaller than or equal to the threshold
    mask = dists <= threshold

    # Filter points and create a red color array for each selected point
    red_points = points[mask]
    red_colors = np.tile(np.array([1.0, 0.0, 0.0]), (red_points.shape[0], 1))
    
    # If normals are available, filter them too
    if pcd_new_legacy.has_normals():
        normals = np.asarray(pcd_new_legacy.normals)
        red_normals = normals[mask]

    # Create a new legacy point cloud for the red points
    pcd_red = o3d.geometry.PointCloud()
    pcd_red.points = o3d.utility.Vector3dVector(red_points)
    pcd_red.colors = o3d.utility.Vector3dVector(red_colors)
    if pcd_new_legacy.has_normals():
        pcd_red.normals = o3d.utility.Vector3dVector(red_normals)

    # Update the visualizer: replace the geometry with the filtered red points
    vis.clear_geometries()
    vis.add_geometry(pcd_red)

    # print(pcd_red)

    # Check if Enter has been pressed; if so, save the red point cloud with an incrementing filename
    if save_trigger.is_set():
        filename = f"tmp/data_{file_num}.pcd"
        o3d.io.write_point_cloud(filename, pcd_red)
        print(f"Saved {filename}")
        file_num += 1
        save_trigger.clear()

    # Update the visualizer window
    vis.poll_events()
    vis.update_renderer()

# Stop sensor capture and close the window (this part is reached if the loop ends)
rs.stop_capture()
vis.destroy_window()
