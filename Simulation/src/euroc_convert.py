import pandas as pd
import numpy as np
import os
import shutil

ROOT = "."
OUT = "./mav0"
os.makedirs(f"{OUT}/cam0/data", exist_ok=True)
os.makedirs(f"{OUT}/imu0", exist_ok=True)

# --- Camera ---
cam = pd.read_csv(f"{ROOT}/camera_front_index.csv")
cam['t_ns'] = (cam['t'] * 1e9).astype(np.int64)

with open(f"{OUT}/cam0/data.csv", 'w') as f:
    f.write("#timestamp [ns],filename\n")
    for _, row in cam.iterrows():
        ts = row['t_ns']
        src = f"{ROOT}/camera_front/{row['filename']}"
        dst = f"{OUT}/cam0/data/{ts}.png"
        if os.path.exists(src):
            shutil.copy2(src, dst)
            f.write(f"{ts},{ts}.png\n")

# --- IMU ---
imu = pd.read_csv(f"{ROOT}/imu.csv")
imu['t_ns'] = (imu['t'] * 1e9).astype(np.int64)

# Trim to camera overlap window
c0, c1 = cam['t_ns'].min(), cam['t_ns'].max()
imu = imu[(imu['t_ns'] >= c0) & (imu['t_ns'] <= c1)]

# Use exact column names from your CSV: wx, wy, wz, ax, ay, az
imu[['t_ns', 'wx', 'wy', 'wz', 'ax', 'ay', 'az']].to_csv(
    f"{OUT}/imu0/data.csv", index=False,
    header=["#timestamp [ns]", "w_x [rad/s]", "w_y [rad/s]", "w_z [rad/s]",
            "a_x [m/s^2]", "a_y [m/s^2]", "a_z [m/s^2]"])

# --- Ground Truth (optional) ---
gt = pd.read_csv(f"{ROOT}/ground_truth.csv")
gt['t_ns'] = (gt['t'] * 1e9).astype(np.int64)
gt = gt[(gt['t_ns'] >= c0) & (gt['t_ns'] <= c1)]
os.makedirs(f"{OUT}/state_groundtruth_estimate0", exist_ok=True)
gt[['t_ns', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']].to_csv(
    f"{OUT}/state_groundtruth_estimate0/data.csv", index=False,
    header=["#timestamp [ns]", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"])

# --- timestamps.txt ---
cam['t_ns'].to_csv(f"{ROOT}/timestamps.txt", index=False, header=False)

print("Done. mav0/ structure created.")
print(f"Camera frames: {len(cam)}")
print(f"IMU samples:   {len(imu)}")
print(f"Ground truth:  {len(gt)}")