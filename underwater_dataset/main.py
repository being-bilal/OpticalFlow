# main.py
# Testing Optical Flow (Stereo-Depth-Based Camera Motion) on NTNU-ARL underwater dataset
# https://huggingface.co/datasets/ntnu-arl/underwater-datasets

import cv2
import numpy as np
import yaml
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from stereo_depth import load_stereo_setup, get_depth
from velocity_calculation import solve_camera_velocity

# ---------------- CONFIG ----------------
BAG_PATH = Path("dataset/mclab.bag")
GT_PATH = Path("dataset/mclab.tum")
INTRINSICS_WATER = Path("dataset/camchain-stereo-intrinsics-underwater.yaml")
EXTRINSICS_AIR = Path("dataset/camchain-imucam-stereo-extrinsics-air.yaml")

CAM0_TOPIC = "/alphasense_driver_ros/cam0"
CAM1_TOPIC = "/alphasense_driver_ros/cam1"
IMU_TOPIC = "/alphasense_driver_ros/imu"

SUBSAMPLE = 8          # pixel stride for velocity least-squares solve (speed vs accuracy tradeoff)
LIMIT_FRAMES = 200     # set to None for the full dataset once this test run looks correct
Z_MIN = 0.1            # meters -- plausible minimum floor/scene distance
Z_MAX = 3.0            # meters -- plausible maximum floor/scene distance for this pool/tank

# ---------------- CALIBRATION ----------------
with open(INTRINSICS_WATER) as f:
    calib = yaml.safe_load(f)
fx, fy, cx, cy = calib['cam0']['intrinsics']

with open(EXTRINSICS_AIR) as f:
    extrinsics_air = yaml.safe_load(f)
R_cam_imu = np.array(extrinsics_air['cam0']['T_cam_imu'])[:3, :3]  # IMU -> camera rotation

stereo_setup = load_stereo_setup(str(INTRINSICS_WATER))

# ---------------- EXTRACT FROM BAG ----------------
images0, images1 = [], []
image_ts0, image_ts1 = [], []
imu_records = []

print("Extracting bag contents...")
with AnyReader([BAG_PATH]) as reader:
    cam0_conns = [c for c in reader.connections if c.topic == CAM0_TOPIC]
    cam1_conns = [c for c in reader.connections if c.topic == CAM1_TOPIC]
    imu_conns = [c for c in reader.connections if c.topic == IMU_TOPIC]

    for conn, ts, raw in reader.messages(connections=cam0_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        images0.append(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width))
        image_ts0.append(ts / 1e9)

    for conn, ts, raw in reader.messages(connections=cam1_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        images1.append(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width))
        image_ts1.append(ts / 1e9)

    for conn, ts, raw in reader.messages(connections=imu_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        # NOTE: orientation field on this topic is invalid (zero quaternion, raw IMU, no fusion).
        # Only angular_velocity is used -- heading is integrated manually in the main loop.
        imu_records.append([ts / 1e9,
                             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

image_ts0 = np.array(image_ts0)
image_ts1 = np.array(image_ts1)
imu_records = np.array(imu_records)
print(f"Extracted: cam0={len(images0)}, cam1={len(images1)}, IMU={len(imu_records)}")

if LIMIT_FRAMES is not None:
    images0 = images0[:LIMIT_FRAMES]
    image_ts0 = image_ts0[:LIMIT_FRAMES]
    print(f"LIMIT_FRAMES active -- using first {LIMIT_FRAMES} cam0 frames only")

# Match cam1 frames to cam0 by nearest timestamp (counts can differ slightly)
cam1_match_idx = np.array([np.argmin(np.abs(image_ts1 - t)) for t in image_ts0])

# ---------------- NORMALIZED PIXEL GRID (built once, reused every frame) ----------------
h, w = images0[0].shape
ys, xs = np.mgrid[0:h, 0:w]
x_norm = (xs - cx) / fx
y_norm = (ys - cy) / fy

# ---------------- MAIN VO LOOP ----------------
x_pos, y_pos, z_pos = 0.0, 0.0, 0.0
heading = 0.0  # gyro-integrated yaw (radians)
trajectory = [(x_pos, y_pos, z_pos)]
vo_timestamps = [image_ts0[0]]
nan_frame_count = 0

print("Computing initial stereo depth...")
rect0_prev, Z_prev = get_depth(images0[0], images1[cam1_match_idx[0]], stereo_setup, Z_MIN, Z_MAX)

N = len(images0)
print(f"Running VO loop over {N - 1} frame pairs...")

for i in range(1, N):
    dt = image_ts0[i] - image_ts0[i - 1]

    rect0_curr, Z_curr = get_depth(images0[i], images1[cam1_match_idx[i]], stereo_setup, Z_MIN, Z_MAX)

    flow = cv2.calcOpticalFlowFarneback(rect0_prev, rect0_curr, None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)

    imu_idx = np.argmin(np.abs(imu_records[:, 0] - image_ts0[i - 1]))
    w_imu = imu_records[imu_idx, 1:4]
    wx_raw, wy_raw, wz_raw = w_imu

    w_cam = R_cam_imu @ w_imu       # rotate IMU angular velocity into camera frame
    wx, wy, wz = w_cam * dt         # rad/s -> rad/frame

    (Vx, Vy, Vz), n_valid = solve_camera_velocity(flow, x_norm, y_norm, fx, fy, wx, wy, wz,
                                                   Z_prev, subsample=SUBSAMPLE)

    if not np.all(np.isfinite([Vx, Vy, Vz])):
        print(f"WARNING: Frame {i} produced non-finite velocity -- treating as zero motion")
        Vx, Vy, Vz = 0.0, 0.0, 0.0
        nan_frame_count += 1

    R_yaw = Rotation.from_euler('z', heading).as_matrix()
    motion_world = R_yaw @ np.array([Vx, Vy, Vz])

    x_pos += motion_world[0]
    y_pos += motion_world[1]
    z_pos += motion_world[2]

    heading += wz_raw * dt          # gyro-integrated yaw (orientation field was invalid)

    trajectory.append((x_pos, y_pos, z_pos))
    vo_timestamps.append(image_ts0[i])

    rect0_prev, Z_prev = rect0_curr, Z_curr

    if i % 50 == 0 or n_valid < 10:
        valid_pixel_count = np.sum(np.isfinite(Z_prev))
        z_min_actual = np.nanmin(Z_prev) if valid_pixel_count > 0 else float('nan')
        z_max_actual = np.nanmax(Z_prev) if valid_pixel_count > 0 else float('nan')
        print(f"Frame {i}/{N-1}: pos=({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f}), "
              f"heading={np.degrees(heading):.1f}°, n_valid_solve_pts={n_valid}, "
              f"valid_Z_pixels={valid_pixel_count}, Z_range=[{z_min_actual:.2f}, {z_max_actual:.2f}]")

print(f"\nTotal frames with non-finite velocity (treated as zero motion): {nan_frame_count}/{N-1}")

trajectory = np.array(trajectory)
vo_timestamps = np.array(vo_timestamps)
print(f"Final VO position: {trajectory[-1]}")

# ---------------- LOAD GROUND TRUTH ----------------
gt_data = np.loadtxt(GT_PATH)
gt_timestamps = gt_data[:, 0]
gt_positions = gt_data[:, 1:4]

matched_gt = np.array([gt_positions[np.argmin(np.abs(gt_timestamps - ts))] for ts in vo_timestamps])

# ---------------- FILTER ANY REMAINING NON-FINITE ROWS (safety net) ----------------
valid_traj = np.all(np.isfinite(trajectory), axis=1)
print(f"Valid trajectory points: {valid_traj.sum()} / {len(trajectory)}")

trajectory_clean = trajectory[valid_traj]
matched_gt_clean = matched_gt[valid_traj]

if len(trajectory_clean) < 5:
    raise RuntimeError("Too few valid trajectory points to align -- check Z_MIN/Z_MAX bounds "
                        "and stereo matching quality before proceeding.")

# ---------------- ALIGN (Umeyama: rotation + translation + scale) ----------------
def umeyama_alignment(source, target):
    mu_src, mu_tgt = source.mean(axis=0), target.mean(axis=0)
    src_c, tgt_c = source - mu_src, target - mu_tgt
    cov = tgt_c.T @ src_c / len(source)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / np.var(src_c, axis=0).sum()
    t = mu_tgt - scale * R @ mu_src
    aligned = (scale * (R @ source.T).T) + t
    return aligned, scale


aligned_vo, scale = umeyama_alignment(trajectory_clean, matched_gt_clean)
print(f"Umeyama scale factor: {scale:.3f} (expect close to 1.0 -- pipeline is metric via stereo depth)")

errors = np.linalg.norm(aligned_vo - matched_gt_clean, axis=1)
print(f"\n--- RESULTS ---")
print(f"ATE RMSE: {np.sqrt(np.mean(errors ** 2)):.3f} m")
print(f"ATE Mean: {np.mean(errors):.3f} m")
print(f"ATE Max:  {np.max(errors):.3f} m")

# ---------------- PLOT ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(matched_gt_clean[:, 0], matched_gt_clean[:, 1], 'g-', linewidth=1.5, label='Ground Truth')
axes[0].plot(aligned_vo[:, 0], aligned_vo[:, 1], 'r--', linewidth=1.2, label='Optical Flow VO')
axes[0].scatter(matched_gt_clean[0, 0], matched_gt_clean[0, 1], c='blue', s=80, marker='o', label='Start', zorder=5)
axes[0].scatter(matched_gt_clean[-1, 0], matched_gt_clean[-1, 1], c='black', s=80, marker='x', label='GT End', zorder=5)
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Y (m)')
axes[0].set_title('Trajectory (Top-Down View)')
axes[0].legend()
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

axes[1].plot(errors)
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Error (m)')
axes[1].set_title('Position Error Over Time (ATE)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vo_result.png', dpi=150)
plt.show()
print("\nSaved plot to vo_result.png")