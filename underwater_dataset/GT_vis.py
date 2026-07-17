import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def load_tum_trajectory(path):
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]      
    quaternions = data[:, 4:8]    
    return timestamps, positions, quaternions

gt_timestamps, gt_positions, gt_quats = load_tum_trajectory("dataset/mclab.tum")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'g-', linewidth=1.5)
ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], c='blue', s=100, label='Start')
ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], c='red', s=100, label='End')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Ground Truth Trajectory (3D)')
ax.legend()
plt.savefig('gt_trajectory_3d.png', dpi=150)
plt.show()