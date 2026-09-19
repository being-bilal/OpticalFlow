import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_tum_file(filename):
    data = np.loadtxt(filename)
    timestamps = data[:, 0]
    positions = data[:, 1:4]  # x, y, z columns
    return timestamps, positions

def plot_tum_trajectories(file1, file2, labels=None):
    """Plot two TUM trajectories"""
    
    # Read both files
    ts1, pos1 = read_tum_file(file1)
    ts2, pos2 = read_tum_file(file2)
    
    labels = labels or [file1.split('/')[-1], file2.split('/')[-1]]
    
    # Color theme: Teal & Coral
    color1 = '#008B8B'  # Dark teal
    color2 = '#FF7F50'  # Coral
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2], color=color1, label=labels[0], linewidth=2)
    ax1.plot(pos2[:, 0], pos2[:, 1], pos2[:, 2], color=color2, label=labels[1], linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectories')
    ax1.legend()
    ax1.grid(True)
    
    # 2D XY plot
    ax2 = fig.add_subplot(122)
    ax2.plot(pos1[:, 0], pos1[:, 1], color=color1, label=labels[0], linewidth=2)
    ax2.plot(pos2[:, 0], pos2[:, 1], color=color2, label=labels[1], linewidth=2, linestyle='--')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('2D Trajectory (XY plane)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Usage
plot_tum_trajectories('gt.tum', 'ekf.tum', 
                      labels=['Ground Truth', 'Estimate (without Alignment)'])