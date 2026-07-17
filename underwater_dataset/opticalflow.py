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

    return flow