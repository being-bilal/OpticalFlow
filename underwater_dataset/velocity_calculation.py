# velocity_calculation.py
import numpy as np


def solve_camera_velocity(flow, x_norm, y_norm, fx, fy, wx, wy, wz, Z_map, subsample=8):
    """
    flow: (H, W, 2) raw pixel optical flow between two frames
    x_norm, y_norm: (H, W) normalized pixel coordinate grids, precomputed once
    fx, fy: focal lengths (pixels)
    wx, wy, wz: angular displacement IN CAMERA FRAME, already scaled to rad/frame (w * dt)
    Z_map: (H, W) per-pixel depth in meters, NaN where invalid
    subsample: pixel stride to reduce point count for speed

    Returns: np.array([Vx, Vy, Vz]) -- per-frame translational displacement in camera frame.
             Returns [0, 0, 0] if too few valid depth points are available this frame.
    """
    ys = np.arange(0, flow.shape[0], subsample)
    xs = np.arange(0, flow.shape[1], subsample)

    u_px = flow[np.ix_(ys, xs)][:, :, 0]
    v_px = flow[np.ix_(ys, xs)][:, :, 1]
    x = x_norm[np.ix_(ys, xs)]
    y = y_norm[np.ix_(ys, xs)]
    Z = Z_map[np.ix_(ys, xs)]

    u_n = u_px / fx
    v_n = v_px / fy

    # Rotational flow component (per-pixel, depends only on angular motion + pixel position)
    u_r = -x * y * wx + (1 + x ** 2) * wy - y * wz
    v_r = -(1 + y ** 2) * wx + x * y * wy + x * wz

    # Pure translational flow = total - rotational
    u_t = (u_n - u_r).ravel()
    v_t = (v_n - v_r).ravel()
    x_flat = x.ravel()
    y_flat = y.ravel()
    Z_flat = Z.ravel()

    # Keep only pixels with valid (non-NaN, positive, in-range) depth
    valid = np.isfinite(Z_flat) & (Z_flat > 0)
    x_flat, y_flat, Z_flat = x_flat[valid], y_flat[valid], Z_flat[valid]
    u_t, v_t = u_t[valid], v_t[valid]

    n = len(x_flat)
    if n < 10:
        return np.array([0.0, 0.0, 0.0]), n  # too few valid points -- treat as no motion this frame

    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)

    A[0::2, 0] = -1
    A[0::2, 2] = x_flat
    b[0::2] = u_t * Z_flat

    A[1::2, 1] = -1
    A[1::2, 2] = y_flat
    b[1::2] = v_t * Z_flat

    V, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    if not np.all(np.isfinite(V)):
        return np.array([0.0, 0.0, 0.0]), n

    return V, n