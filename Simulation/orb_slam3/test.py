#!/usr/bin/env python3
"""
Plot trajectories with time-sync and Umeyama alignment.
Creates 3 comparison sets: one for each reference (GT, EKF, GTSAM).
VIO trajectories (.txt) are aligned to world frame using Umeyama.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_DIR = '/Users/mohammadbilal/Documents/Projects/OpticalFlow/Simulation/orb_slam3/results'

# Load TUM format .txt files
def load_tum(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            ts_sec = float(p[0]) / 1e9
            rows.append([ts_sec] + [float(v) for v in p[1:8]])
    return pd.DataFrame(rows, columns=['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

# Load CSV files
def load_csv(path):
    df = pd.read_csv(path)
    return df[['t', 'x', 'y', 'z']].copy()

# Umeyama alignment: transform src to align with dst
def umeyama_alignment(src, dst, with_scale=True):
    """Rigid + scale alignment: dst ≈ s * R @ src + t"""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    
    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / len(src)
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t

def apply_transform(points, s, R, t):
    """Apply: p_new = s * R @ p_old + t"""
    return (s * (R @ points.T)).T + t

# Time-sync: for each src timestamp, find closest ref timestamp
def sync_to_reference(src_t, ref_t, max_dt=0.05):
    """Match each src timestamp to closest ref timestamp"""
    src_idx = []
    ref_idx = []
    
    for i, t_src in enumerate(src_t):
        diffs = np.abs(ref_t - t_src)
        j = np.argmin(diffs)
        
        if diffs[j] <= max_dt:
            src_idx.append(i)
            ref_idx.append(j)
    
    return np.array(src_idx), np.array(ref_idx)

# Load all files
print("Loading trajectories...")
cam = load_tum(f'{DATA_DIR}/CameraTrajectory.txt')
kf = load_tum(f'{DATA_DIR}/KeyFrameTrajectory.txt')
ekf = load_csv(f'{DATA_DIR}/ekf.csv')
gtsam = load_csv(f'{DATA_DIR}/gtsam.csv')
gt = load_csv(f'{DATA_DIR}/ground_truth.csv')

print(f"Camera: {len(cam)}, Keyframe: {len(kf)}, EKF: {len(ekf)}, GTSAM: {len(gtsam)}, GT: {len(gt)}")

# Function to process and plot for a given reference
def process_reference(ref_name, reference_df):
    print(f"\n{'='*60}")
    print(f"Processing with {ref_name} as reference")
    print(f"{'='*60}")
    
    ref_t = reference_df['t'].values
    ref_xyz = reference_df[['x', 'y', 'z']].values
    
    # Time-sync all trajectories to reference
    cam_idx, ref_idx_cam = sync_to_reference(cam['t'].values, ref_t, max_dt=0.05)
    kf_idx, ref_idx_kf = sync_to_reference(kf['t'].values, ref_t, max_dt=0.05)
    ekf_idx, ref_idx_ekf = sync_to_reference(ekf['t'].values, ref_t, max_dt=0.05)
    gtsam_idx, ref_idx_gtsam = sync_to_reference(gtsam['t'].values, ref_t, max_dt=0.05)
    
    print(f"Matched frames - Camera: {len(cam_idx)}, Keyframe: {len(kf_idx)}, EKF: {len(ekf_idx)}, GTSAM: {len(gtsam_idx)}")
    
    # Extract matched poses
    cam_xyz_raw = cam.iloc[cam_idx][['x', 'y', 'z']].values
    kf_xyz_raw = kf.iloc[kf_idx][['x', 'y', 'z']].values
    ekf_xyz = ekf.iloc[ekf_idx][['x', 'y', 'z']].values
    gtsam_xyz = gtsam.iloc[gtsam_idx][['x', 'y', 'z']].values
    ref_xyz_matched = ref_xyz[ref_idx_cam]  # Use cam indices for reference
    
    # Align VIO trajectories (Camera & Keyframe) to reference frame
    print(f"Aligning Camera to {ref_name}...")
    s_cam, R_cam, t_cam = umeyama_alignment(cam_xyz_raw, ref_xyz_matched, with_scale=True)
    cam_xyz_aligned = apply_transform(cam.iloc[cam_idx][['x', 'y', 'z']].values, s_cam, R_cam, t_cam)
    print(f"  Scale: {s_cam:.4f}")
    
    print(f"Aligning Keyframes to {ref_name}...")
    ref_xyz_kf = ref_xyz[ref_idx_kf]
    s_kf, R_kf, t_kf = umeyama_alignment(kf_xyz_raw, ref_xyz_kf, with_scale=True)
    kf_xyz_aligned = apply_transform(kf.iloc[kf_idx][['x', 'y', 'z']].values, s_kf, R_kf, t_kf)
    print(f"  Scale: {s_kf:.4f}")
    
    # For non-reference CSV files, align them too if they're not the reference
    if ref_name != 'EKF':
        print(f"Aligning EKF to {ref_name}...")
        ref_xyz_ekf = ref_xyz[ref_idx_ekf]
        s_ekf, R_ekf, t_ekf = umeyama_alignment(ekf_xyz, ref_xyz_ekf, with_scale=True)
        ekf_xyz = apply_transform(ekf_xyz, s_ekf, R_ekf, t_ekf)
        print(f"  Scale: {s_ekf:.4f}")
    
    if ref_name != 'GTSAM':
        print(f"Aligning GTSAM to {ref_name}...")
        ref_xyz_gtsam = ref_xyz[ref_idx_gtsam]
        s_gtsam, R_gtsam, t_gtsam = umeyama_alignment(gtsam_xyz, ref_xyz_gtsam, with_scale=True)
        gtsam_xyz = apply_transform(gtsam_xyz, s_gtsam, R_gtsam, t_gtsam)
        print(f"  Scale: {s_gtsam:.4f}")
    
    # Get timestamps for plotting
    t_cam_plot = cam.iloc[cam_idx]['t'].values
    t_kf_plot = kf.iloc[kf_idx]['t'].values
    t_ekf_plot = ekf.iloc[ekf_idx]['t'].values
    t_gtsam_plot = gtsam.iloc[gtsam_idx]['t'].values
    t_ref_plot = reference_df.iloc[ref_idx_cam]['t'].values
    
    # Prepare data dict for plotting
    trajectories = {
        ref_name: {'xyz': ref_xyz_matched, 't': t_ref_plot, 'color': 'black', 'width': 2.5, 'label': ref_name},
        'Camera': {'xyz': cam_xyz_aligned, 't': t_cam_plot, 'color': 'blue', 'width': 1.5, 'label': 'Camera'},
        'Keyframes': {'xyz': kf_xyz_aligned, 't': t_kf_plot, 'color': 'cyan', 'width': 1.5, 'label': 'Keyframes', 'marker': 'o'},
    }
    
    if ref_name != 'EKF':
        trajectories['EKF'] = {'xyz': ekf_xyz, 't': t_ekf_plot, 'color': 'orange', 'width': 1.5, 'label': 'EKF'}
    
    if ref_name != 'GTSAM':
        trajectories['GTSAM'] = {'xyz': gtsam_xyz, 't': t_gtsam_plot, 'color': 'green', 'width': 1.5, 'label': 'GTSAM'}
    
    # 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for name, data in trajectories.items():
        ax.plot(data['xyz'][:, 0], data['xyz'][:, 1], data['xyz'][:, 2], 
                label=data['label'], color=data['color'], linewidth=data['width'], alpha=0.8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Trajectory (Aligned to {ref_name})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{DATA_DIR}/plot_3d_vs_{ref_name.lower()}.png', dpi=150)
    
    # Top-down plot
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    for name, data in trajectories.items():
        ax2.plot(data['xyz'][:, 0], data['xyz'][:, 1], 
                label=data['label'], color=data['color'], linewidth=data['width'], alpha=0.8)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Top-down View (Aligned to {ref_name})')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f'{DATA_DIR}/plot_topdown_vs_{ref_name.lower()}.png', dpi=150)
    
    # Z vs Time plot
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    t0 = t_ref_plot[0]
    for name, data in trajectories.items():
        ax3.plot(data['t'] - t0, data['xyz'][:, 2], 
                label=data['label'], color=data['color'], linewidth=data['width'], alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title(f'Depth (Z) vs Time (Aligned to {ref_name})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(f'{DATA_DIR}/plot_depth_vs_{ref_name.lower()}.png', dpi=150)
    
    print(f"Saved plots for {ref_name} reference")

# Process each reference
process_reference('Ground_Truth', gt)
process_reference('EKF', ekf)
process_reference('GTSAM', gtsam)

print(f"\n{'='*60}")
print("All plots saved!")
print(f"{'='*60}")