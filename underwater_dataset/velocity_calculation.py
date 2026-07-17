import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def solve_camera_velocity(flow, x_norm, y_norm, fx, fy, wx, wy, wz, Z, subsample=8):
    ys = np.arange(0, flow.shape[0], subsample)
    xs = np.arange(0, flow.shape[1], subsample)

    u_px = flow[np.ix_(ys, xs)][:, :, 0]
    v_px = flow[np.ix_(ys, xs)][:, :, 1]
    x = x_norm[np.ix_(ys, xs)]
    y = y_norm[np.ix_(ys, xs)]

    u_n = u_px / fx
    v_n = v_px / fy

    u_r = -x * y * wx + (1 + x**2) * wy - y * wz
    v_r = -(1 + y**2) * wx + x * y * wy + x * wz

    u_t = (u_n - u_r).ravel()
    v_t = (v_n - v_r).ravel()
    x_flat = x.ravel()
    y_flat = y.ravel()

    n = len(x_flat)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)

    A[0::2, 0] = -1
    A[0::2, 2] = x_flat
    b[0::2] = u_t * Z

    A[1::2, 1] = -1
    A[1::2, 2] = y_flat
    b[1::2] = v_t * Z

    V, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return V  # Vx, Vy, Vz