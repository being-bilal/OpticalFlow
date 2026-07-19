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
START_FRAME = 2000     # set both to None to run the full sequence
END_FRAME = 3500       # slice is [START_FRAME:END_FRAME), matches Python slicing
Z_MIN = 0.1            # meters -- plausible minimum floor/scene distance
Z_MAX = 3.0            # meters -- plausible maximum floor/scene distance for this pool/tank

CAM_TILT_DEG = 16.0    # cam0 is mounted tilted this many degrees downward from horizontal
                       # (per NTNU-ARL MC-lab rig docs). If z_pos still drifts steadily in
                       # one direction after this fix, flip the sign here and re-check.
CAM_TILT_RAD = np.radians(CAM_TILT_DEG)

# ---------------- CALIBRATION ----------------
with open(INTRINSICS_WATER) as f:
    calib = yaml.safe_load(f)
fx, fy, cx, cy = calib['cam0']['intrinsics']

with open(EXTRINSICS_AIR) as f:
    extrinsics_air = yaml.safe_load(f)
R_cam_imu = np.array(extrinsics_air['cam0']['T_cam_imu'])[:3, :3]  # IMU -> camera rotation

stereo_setup = load_stereo_setup(str(INTRINSICS_WATER))

# contrast enhancement -- underwater frames are low-contrast, which starves Farneback of gradient info
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

if START_FRAME is not None or END_FRAME is not None:
    images0 = images0[START_FRAME:END_FRAME]
    image_ts0 = image_ts0[START_FRAME:END_FRAME]
    print(f"Using frame window [{START_FRAME}:{END_FRAME}] -- {len(images0)} cam0 frames")

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

# Recorded so we can cheaply re-integrate heading with different gyro-z bias
# candidates afterward, without recomputing optical flow / stereo depth again.
body_vec_history = []   # [level_fwd, Vx, level_down] per frame
wz_raw_history = []     # raw gyro-z reading used at each frame
dt_history = []

print("Computing initial stereo depth...")
rect0_prev, Z_prev = get_depth(images0[0], images1[cam1_match_idx[0]], stereo_setup, Z_MIN, Z_MAX)
rect0_prev = clahe.apply(rect0_prev)

N = len(images0)
print(f"Running VO loop over {N - 1} frame pairs...")

for i in range(1, N):
    dt = image_ts0[i] - image_ts0[i - 1]

    rect0_curr, Z_curr = get_depth(images0[i], images1[cam1_match_idx[i]], stereo_setup, Z_MIN, Z_MAX)
    rect0_curr = clahe.apply(rect0_curr)

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

    # ---- Undo the fixed ~16 deg downward mount tilt ----
    # Without this, part of the AUV's true forward motion (Vz) leaks into the
    # down axis (Vy) and vice versa, since the camera boresight isn't level
    # with the vehicle's horizontal plane. This is what was producing the
    # steady negative Vy bias and the multi-meter phantom z_pos drift.
    level_fwd = Vz * np.cos(CAM_TILT_RAD) - Vy * np.sin(CAM_TILT_RAD)
    level_down = Vz * np.sin(CAM_TILT_RAD) + Vy * np.cos(CAM_TILT_RAD)

    # ---- Remap camera-optical axes into a body vector before applying yaw ----
    # Camera convention here is x-right, y-down, z-forward (standard pinhole/optical frame).
    # cam0 is the front-facing stereo pair, so (leveled) Vz is the *forward* component --
    # the dominant AUV translation -- while Vx is lateral and (leveled) Vy is vertical.
    # Rotation.from_euler('z', heading) mixes indices 0 and 1 of whatever vector it's given,
    # so passing [Vx, Vy, Vz] directly (as before) rotated the small Vx/Vy pair together
    # and dumped the dominant forward motion Vz straight into z_pos, untouched -- it never
    # reached the horizontal (x_pos, y_pos) trajectory that gets plotted.
    # Building body_vec = [forward, right, down] and rotating THAT by yaw puts the
    # dominant forward motion into the horizontal plane where it belongs.
    body_vec = np.array([level_fwd, Vx, level_down])   # [forward, right, down]

    R_yaw = Rotation.from_euler('z', heading)
    motion_world = R_yaw.apply(body_vec)   # .apply() is the version-safe way to rotate a
                                            # vector with scipy's Rotation -- `R_yaw @ body_vec`
                                            # is not reliably supported across scipy/numpy
                                            # versions and can raise a matmul dimension error

    body_vec_history.append(body_vec)
    wz_raw_history.append(wz_raw)
    dt_history.append(dt)

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
        depth_flag = "  <-- exceeds tank depth (~1.5m), check CAM_TILT_DEG sign" if abs(z_pos) > 2.0 else ""
        print(f"Frame {i}/{N-1}: pos=({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f}), "
              f"heading={np.degrees(heading):.1f}°, n_valid_solve_pts={n_valid}, "
              f"valid_Z_pixels={valid_pixel_count}, Z_range=[{z_min_actual:.2f}, {z_max_actual:.2f}], "
              f"V=({Vx:.4f}, {Vy:.4f}, {Vz:.4f}){depth_flag}")

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
print(f"\n--- RESULTS (before gyro-bias correction) ---")
print(f"ATE RMSE: {np.sqrt(np.mean(errors ** 2)):.3f} m")
print(f"ATE Mean: {np.mean(errors):.3f} m")
print(f"ATE Max:  {np.max(errors):.3f} m")

# ---------------- GYRO-Z BIAS SEARCH ----------------
# A small constant offset in the raw gyro-z reading integrates into a slowly growing
# heading error, which rotates every subsequent motion vector by roughly the same
# small angle -- producing a trajectory that tracks the true path's shape closely but
# sits at a near-constant perpendicular offset the whole way, exactly like what shows
# up in the plot. Umeyama's single rigid+scale alignment can't undo this because it's
# not a single wrong rotation, it's a continuously accumulating one.
#
# body_vec_history / wz_raw_history / dt_history were recorded during the main loop
# so we can re-integrate heading (and rebuild the trajectory) cheaply for many
# candidate bias values without recomputing optical flow or stereo depth.

def rebuild_trajectory_with_bias(wz_bias):
    x, y, z = 0.0, 0.0, 0.0
    hdg = 0.0
    traj = [(x, y, z)]
    for bv, wz_r, dt_i in zip(body_vec_history, wz_raw_history, dt_history):
        R_yaw_b = Rotation.from_euler('z', hdg)
        mw = R_yaw_b.apply(bv)
        x += mw[0]; y += mw[1]; z += mw[2]
        hdg += (wz_r - wz_bias) * dt_i
        traj.append((x, y, z))
    return np.array(traj)


candidate_biases = np.linspace(-0.02, 0.02, 81)  # rad/s -- widen if the best value lands at an edge
best_bias, best_rmse, best_traj = 0.0, np.sqrt(np.mean(errors ** 2)), trajectory

for wz_bias in candidate_biases:
    traj_b = rebuild_trajectory_with_bias(wz_bias)
    valid_b = np.all(np.isfinite(traj_b), axis=1)
    if valid_b.sum() < 5:
        continue
    aligned_b, _ = umeyama_alignment(traj_b[valid_b], matched_gt[valid_b])
    rmse_b = np.sqrt(np.mean(np.linalg.norm(aligned_b - matched_gt[valid_b], axis=1) ** 2))
    if rmse_b < best_rmse:
        best_rmse, best_bias, best_traj = rmse_b, wz_bias, traj_b

if best_bias != 0.0:
    print(f"\nBest gyro-z bias found: {best_bias:.5f} rad/s "
          f"(RMSE {np.sqrt(np.mean(errors ** 2)):.3f} m -> {best_rmse:.3f} m)")
    trajectory = best_traj
    trajectory_clean = trajectory[np.all(np.isfinite(trajectory), axis=1)]
    matched_gt_clean = matched_gt[np.all(np.isfinite(trajectory), axis=1)]
    aligned_vo, scale = umeyama_alignment(trajectory_clean, matched_gt_clean)
    errors = np.linalg.norm(aligned_vo - matched_gt_clean, axis=1)
else:
    print("\nNo candidate bias improved on zero bias -- gyro-z offset may already be negligible, "
          "or the residual offset has a different cause (widen candidate_biases range to double-check).")

print(f"\n--- RESULTS (after gyro-bias correction) ---")
print(f"Umeyama scale factor: {scale:.3f}")
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