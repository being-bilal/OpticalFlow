# velocity_calculation.py
import numpy as np


def solve_camera_velocity(flow, x_norm, y_norm, fx, fy, wx, wy, wz, Z_map, subsample=8,
                           ransac_iters=200, ransac_thresh_px=2.0, min_inliers=10, rng=None):
    """
    flow: (H, W, 2) raw pixel optical flow between two frames
    x_norm, y_norm: (H, W) normalized pixel coordinate grids, precomputed once
    fx, fy: focal lengths (pixels)
    wx, wy, wz: angular displacement IN CAMERA FRAME, already scaled to rad/frame (w * dt)
    Z_map: (H, W) per-pixel depth in meters, NaN where invalid
    subsample: pixel stride to reduce point count for speed
    ransac_iters: number of RANSAC hypothesis trials
    ransac_thresh_px: inlier threshold, expressed in pixels (converted internally to
                       normalized-flow-error units via fx) -- how far a point's residual
                       flow can be from the fitted model and still count as an inlier
    min_inliers: minimum consensus set size to accept the RANSAC result
    rng: optional np.random.Generator for reproducibility; a fresh one is created if None

    Returns: np.array([Vx, Vy, Vz]) -- per-frame translational displacement in camera frame.
             Returns [0, 0, 0] if too few valid depth points, or no RANSAC hypothesis reaches
             min_inliers, are available this frame.
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
    # Standard Longuet-Higgins & Prazdny motion field model:
    #   u_rot =  x*y*wx - (1+x^2)*wy + y*wz
    #   v_rot =  (1+y^2)*wx - x*y*wy - x*wz
    u_r = x * y * wx - (1 + x ** 2) * wy + y * wz
    v_r = (1 + y ** 2) * wx - x * y * wy - x * wz

    # Pure translational flow = total - rotational
    u_t = (u_n - u_r).ravel()
    v_t = (v_n - v_r).ravel()
    x_flat = x.ravel()
    y_flat = y.ravel()
    Z_flat = Z.ravel()

    # Keep only pixels with valid (non-NaN, positive, in-range) depth and finite flow
    valid = np.isfinite(Z_flat) & (Z_flat > 0) & np.isfinite(u_t) & np.isfinite(v_t)
    x_flat, y_flat, Z_flat = x_flat[valid], y_flat[valid], Z_flat[valid]
    u_t, v_t = u_t[valid], v_t[valid]

    n = len(x_flat)
    if n < 10:
        return np.array([0.0, 0.0, 0.0]), 0  # too few valid points -- treat as no motion this frame

    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)

    A[0::2, 0] = -1
    A[0::2, 2] = x_flat
    b[0::2] = u_t * Z_flat

    A[1::2, 1] = -1
    A[1::2, 2] = y_flat
    b[1::2] = v_t * Z_flat

    # ---- RANSAC outlier rejection ----
    # Fit the SAME physical model (translation-only flow given known depth) that the
    # final least-squares solve uses, rather than a generic 2D affine/homography motion
    # model -- a translating camera's flow field isn't globally affine (it depends on
    # per-pixel depth via 1/Z and has a projective x*Vz / y*Vz term), so a generic affine
    # RANSAC would reject good points and keep bad ones. Each point contributes 2 rows to
    # A/b; 2 points give 4 equations for the 3 unknowns (Vx, Vy, Vz), which is enough to
    # be well-determined for most point configurations, so we sample minimal sets of 2
    # points per hypothesis.
    if rng is None:
        rng = np.random.default_rng()

    ransac_thresh = ransac_thresh_px / fx  # convert pixel threshold to normalized-flow units

    min_samples = 2
    best_inliers = None
    best_count = -1

    if n >= min_samples:
        for _ in range(ransac_iters):
            sample_idx = rng.choice(n, size=min_samples, replace=False)
            rows = np.empty(min_samples * 2, dtype=int)
            rows[0::2] = sample_idx * 2
            rows[1::2] = sample_idx * 2 + 1

            V_candidate, _, rank, _ = np.linalg.lstsq(A[rows], b[rows], rcond=None)
            if rank < 3 or not np.all(np.isfinite(V_candidate)):
                continue  # degenerate sample (e.g. collinear/coincident points) -- skip

            resid = (A @ V_candidate - b).reshape(n, 2)
            point_err = np.linalg.norm(resid, axis=1) / np.maximum(Z_flat, 1e-6)
            inlier_mask = point_err < ransac_thresh
            count = int(inlier_mask.sum())

            if count > best_count:
                best_count = count
                best_inliers = inlier_mask

    if best_inliers is None or best_count < min_inliers:
        return np.array([0.0, 0.0, 0.0]), 0  # no RANSAC hypothesis found a usable consensus set

    # Final least-squares refit on the full RANSAC consensus set (more accurate than any
    # single minimal-sample estimate, since it uses every inlier rather than just 2 points)
    idx = np.where(best_inliers)[0]
    rows = np.empty(len(idx) * 2, dtype=int)
    rows[0::2] = idx * 2
    rows[1::2] = idx * 2 + 1

    V, _, _, _ = np.linalg.lstsq(A[rows], b[rows], rcond=None)
    if not np.all(np.isfinite(V)):
        return np.array([0.0, 0.0, 0.0]), 0

    n_final = int(best_count)
    if n_final < min_inliers:
        return np.array([0.0, 0.0, 0.0]), 0

    return V, n_final