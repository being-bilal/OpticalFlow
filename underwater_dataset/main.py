# Testing Optical Flow on underwater dataset 
# https://huggingface.co/datasets/ntnu-arl/underwater-datasets

from rosbags.highlevel import AnyReader
from pathlib import Path
import yaml
import numpy as np
from opticalflow import compute_uv_dense


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




# Compute dense optical flow between consecutive frames
# u_vals, v_vals = compute_uv_dense(images)  
# print(f"u range: [{u_vals.min():.3f}, {u_vals.max():.3f}] px")
# print(f"v range: [{v_vals.min():.3f}, {v_vals.max():.3f}] px")


