# stereo_depth.py
import cv2
import numpy as np
import yaml


def load_stereo_setup(intrinsics_path):
    with open(intrinsics_path) as f:
        calib = yaml.safe_load(f)

    fx0, fy0, cx0, cy0 = calib['cam0']['intrinsics']
    dist0 = np.array(calib['cam0']['distortion_coeffs']).reshape(4, 1)

    fx1, fy1, cx1, cy1 = calib['cam1']['intrinsics']
    dist1 = np.array(calib['cam1']['distortion_coeffs']).reshape(4, 1)

    K0 = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]])
    K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])

    T_cn_cnm1 = np.array(calib['cam1']['T_cn_cnm1'])
    R = T_cn_cnm1[:3, :3]
    T = T_cn_cnm1[:3, 3]
    baseline = np.linalg.norm(T)

    w, h = calib['cam0']['resolution']

    R0, R1, P0, P1, Q = cv2.fisheye.stereoRectify(
        K0, dist0, K1, dist1, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0
    )
    map0x, map0y = cv2.fisheye.initUndistortRectifyMap(K0, dist0, R0, P0, (w, h), cv2.CV_32FC1)
    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(K1, dist1, R1, P1, (w, h), cv2.CV_32FC1)

    f_rectified = P0[0, 0]

    stereo_matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=7,
        P1=8 * 7 ** 2, P2=32 * 7 ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32
    )

    setup = {
        'map0x': map0x, 'map0y': map0y,
        'map1x': map1x, 'map1y': map1y,
        'f': f_rectified, 'baseline': baseline,
        'matcher': stereo_matcher
    }
    print(f"Stereo setup loaded. Baseline={baseline:.4f} m, f_rectified={f_rectified:.2f} px")
    return setup


def get_depth(img0, img1, setup, z_min=0.1, z_max=3.0):
    """
    img0, img1: raw grayscale images from cam0, cam1 (same timestamp)
    setup: dict returned by load_stereo_setup()
    z_min, z_max: plausible depth bounds (meters) -- anything outside is set to NaN
    Returns: Z (H, W) depth map in meters (NaN where invalid), rect0 (rectified cam0 image)
    """
    rect0 = cv2.remap(img0, setup['map0x'], setup['map0y'], cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, setup['map1x'], setup['map1y'], cv2.INTER_LINEAR)

    disparity = setup['matcher'].compute(rect0, rect1).astype(np.float32) / 16.0

    Z = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = disparity > 0
    Z[valid] = (setup['f'] * setup['baseline']) / disparity[valid]

    # Reject implausible depths (too close, too far, or numerically unstable from tiny disparity)
    Z[(Z < z_min) | (Z > z_max)] = np.nan
    return rect0, Z
