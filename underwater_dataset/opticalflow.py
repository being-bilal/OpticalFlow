import cv2
import numpy as np

def compute_uv_dense(images):
    u_vals, v_vals = [], []

    for i in range(len(images) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            images[i], images[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        u_vals.append(u)
        v_vals.append(v)

    return np.array(u_vals), np.array(v_vals)