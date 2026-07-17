# Testing Optical Flow on underwater dataset 
# https://huggingface.co/datasets/ntnu-arl/underwater-datasets

from rosbags.highlevel import AnyReader
from pathlib import Path
import yaml
import numpy as np
from opticalflow import compute_uv_dense
from velocity_calculation import solve_camera_velocity
from scipy.spatial.transform import Rotation


bag_path = Path("dataset/mclab.bag")  
gd_path = Path("dataset/mclab.tum")
cam_intrinsics = Path("dataset/camchain-stereo-intrinsics-underwater.yaml")
CAM_TOPIC = "/alphasense_driver_ros/cam0"
IMU_TOPIC = "/alphasense_driver_ros/imu"           
PRESSURE_TOPIC = "/mavros/imu/static_pressure"     
RANGE_TOPIC = "/mavros/rangefinder/rangefinder"    

# Camera Calibration Parameters
with open(cam_intrinsics) as f:
    calib = yaml.safe_load(f)

images, image_ts = [], []
imu_records = []
range_records = []

# Reading the ROS bag file and extracting messages from topics
with AnyReader([bag_path]) as reader:
    cam_conns = [c for c in reader.connections if c.topic == CAM_TOPIC]
    imu_conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
    range_conns = [c for c in reader.connections if c.topic == RANGE_TOPIC]

    # Camera 
    for conn, ts, raw in reader.messages(connections=cam_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        images.append(img)
        image_ts.append(ts / 1e9)

    # IMU
    for conn, ts, raw in reader.messages(connections=imu_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        imu_records.append([ts / 1e9,
                     msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                     msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    # Depth Sensor 
    for conn, ts, raw in reader.messages(connections=range_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        range_records.append([ts / 1e9, msg.range])  

image_ts = np.array(image_ts)
imu_records = np.array(imu_records)
range_records = np.array(range_records)

print(f"Extracted: {len(images)} images, {len(imu_records)} IMU, {len(range_records)} range readings")

# Converting the pixel coordinates to camera coordinates 
fx, fy, cx, cy = calib['cam0']['intrinsics'] 
h, w = images[0].shape
y, x = np.mgrid[0:h, 0:w] # x and y pixel coordinate matrices
x = (x - cx) / fx
y = (y - cy) / fy

x_pos, y_pos, z_pos = 0.0, 0.0, 0.0
heading = 0.0
trajectory = []


# converting flow to camera coordinates
for i in range(len(images) - 1):
    dt = image_ts[i + 1] - image_ts[i]
    
    # Compute dense optical flow between consecutive frames
    flow = compute_uv_dense(images)  
    imu_idx = np.argmin(np.abs(imu_records[:, 0] - image_ts[i]))
    wx_raw, wy_raw, wz_raw = imu_records[imu_idx, 1:4]
    w_imu = np.array([wx_raw, wy_raw, wz_raw]) # Array of angular velocity from IMU
    
    ### Issue : Assuming camera and IMU are aligned (which is not the case)
    w_cam = w_imu  
    wx, wy, wz = w_cam * dt # rad/s -> rad/frame
    r_idx = np.argmin(np.abs(range_records[:, 0] - image_ts[i]))
    
    ### Issue : Rangefinder is not aligned with camera (rangefinder is facing downwards, camera is facing forward)
    Z = np.clip(range_records[r_idx, 1], 0.1, 3.0)  
    Vx, Vy, Vz = solve_camera_velocity(flow, x, y, fx, fy, wx, wy, wz, Z)
    R_yaw = Rotation.from_euler('z', heading).as_matrix()
    motion_world = R_yaw @ np.array([Vx, Vy, Vz])
    
    x_pos += motion_world[0]
    y_pos += motion_world[1]
    
    qx, qy, qz, qw = imu_records[imu_idx, 4:8]
    heading = Rotation.from_quat([qx, qy, qz, qw]).as_euler('xyz')[2]
    
    trajectory.append((x_pos, y_pos, z_pos))
    
    if i % 500 == 0:
        print(f"Frame {i}/{len(images)-1}: Vx={Vx:.4f}, Vy={Vy:.4f}, Vz={Vz:.4f}")

trajectory = np.array(trajectory)
print(f"Final position: {trajectory[-1]}")
    


