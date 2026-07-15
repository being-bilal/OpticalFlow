# Testing Optical Flow on underwater dataset 
# https://huggingface.co/datasets/ntnu-arl/underwater-datasets
from rosbags.highlevel import AnyReader
from pathlib import Path
import yaml
import numpy as np

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

with AnyReader([bag_path]) as reader:
    cam_conns = [c for c in reader.connections if c.topic == CAM_TOPIC]
    imu_conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
    range_conns = [c for c in reader.connections if c.topic == RANGE_TOPIC]

    for conn, ts, raw in reader.messages(connections=cam_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        images.append(img)
        image_ts.append(ts / 1e9)

    for conn, ts, raw in reader.messages(connections=imu_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        imu_records.append([ts / 1e9,
                             msg.orientation.x, msg.orientation.y,
                             msg.orientation.z, msg.orientation.w])

    for conn, ts, raw in reader.messages(connections=range_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        range_records.append([ts / 1e9, msg.range])  

image_ts = np.array(image_ts)
imu_records = np.array(imu_records)
range_records = np.array(range_records)

print(f"Extracted: {len(images)} images, {len(imu_records)} IMU, {len(range_records)} range readings")
print(f"Sample range values: {range_records[:5, 1]}")  # sanity check — should be plausible meters, e.g. 0.5-2.0