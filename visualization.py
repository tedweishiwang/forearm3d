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

for fid in range(150):
    im_rgbd = rs.capture_frame(True, True)  # RGBD image (Tensor)

    # Build intrinsics as Tensor (you can customize if needed)
    intrinsics = o3d.core.Tensor([
        [615.0, 0.0, 320.0],
        [0.0, 615.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=o3d.core.Dtype.Float32)

    # Identity extrinsic
    extrinsics = o3d.core.Tensor(np.eye(4), dtype=o3d.core.Dtype.Float32)

    # Create point cloud (Tensor API)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        im_rgbd, intrinsics, extrinsics
    )

    # Convert to legacy for visualization
    pcd_legacy = pcd.to_legacy()

    # Flip point cloud for correct orientation
    pcd_legacy.transform([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])

    o3d.visualization.draw_geometries_with_editing([pcd_legacy])

rs.stop_capture()
