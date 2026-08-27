#!/usr/bin/env python3
"""Convert ORB-SLAM3 .txt trajectory files to TUM format with comment header"""

def txt_to_tum(in_path, out_path):
    """Convert TUM format .txt to .tum with header"""
    rows = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Timestamp is in nanoseconds, convert to seconds
            ts_sec = float(parts[0]) / 1e9
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            rows.append((ts_sec, x, y, z, qx, qy, qz, qw))
    
    # Sort by timestamp
    rows.sort(key=lambda r: r[0])
    
    with open(out_path, 'w') as f:
        f.write(f"# {in_path} -> TUM (timestamp tx ty tz qx qy qz qw)\n")
        for t, x, y, z, qx, qy, qz, qw in rows:
            f.write(f"{t:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
    
    print(f"wrote {len(rows)} poses -> {out_path}")


if __name__ == '__main__':
    txt_to_tum('/Users/mohammadbilal/Documents/Projects/OpticalFlow/Simulation/orb_slam3/results/CameraTrajectory.txt', '/Users/mohammadbilal/Documents/Projects/OpticalFlow/Simulation/orb_slam3/results/CameraTrajectory.tum')
    txt_to_tum('/Users/mohammadbilal/Documents/Projects/OpticalFlow/Simulation/orb_slam3/results/KeyFrameTrajectory.txt', '/Users/mohammadbilal/Documents/Projects/OpticalFlow/Simulation/orb_slam3/results/KeyFrameTrajectory.tum')