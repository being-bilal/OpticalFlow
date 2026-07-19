# test_stereo_depth.py
# Diagnostic: sanity-check the stereo depth pipeline in isolation, without the VO loop.
# Checks rectification/epipolar alignment, disparity validity, depth validity, and
# image texture on a handful of sample frames spread across the sequence.

import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
import matplotlib.pyplot as plt

from stereo_depth import load_stereo_setup, get_depth

BAG_PATH = Path("dataset/mclab.bag")
INTRINSICS_WATER = Path("dataset/camchain-stereo-intrinsics-underwater.yaml")
CAM0_TOPIC = "/alphasense_driver_ros/cam0"
CAM1_TOPIC = "/alphasense_driver_ros/cam1"

SAMPLE_FRAME_INDICES = [0, 2000, 2750, 4000, 6000, 7800]   # spread across the sequence
Z_MIN = 0.1
Z_MAX = 3.0

stereo_setup = load_stereo_setup(str(INTRINSICS_WATER))

# ---------------- LOAD ALL FRAMES (needed for cam1 nearest-timestamp match) ----------------
images0, images1 = [], []
image_ts0, image_ts1 = [], []

print("Scanning bag for cam0/cam1 frames...")
with AnyReader([BAG_PATH]) as reader:
    cam0_conns = [c for c in reader.connections if c.topic == CAM0_TOPIC]
    cam1_conns = [c for c in reader.connections if c.topic == CAM1_TOPIC]

    for conn, ts, raw in reader.messages(connections=cam0_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        images0.append(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width))
        image_ts0.append(ts / 1e9)

    for conn, ts, raw in reader.messages(connections=cam1_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        images1.append(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width))
        image_ts1.append(ts / 1e9)

image_ts0 = np.array(image_ts0)
image_ts1 = np.array(image_ts1)
print(f"Loaded cam0={len(images0)}, cam1={len(images1)}")

# ---------------- RUN DIAGNOSTICS ON SAMPLE FRAMES ----------------
for idx in SAMPLE_FRAME_INDICES:
    if idx >= len(images0):
        print(f"Skipping index {idx} -- out of range")
        continue

    ts = image_ts0[idx]
    cam1_idx = np.argmin(np.abs(image_ts1 - ts))

    img0 = images0[idx]
    img1 = images1[cam1_idx]

    rect0 = cv2.remap(img0, stereo_setup['map0x'], stereo_setup['map0y'], cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, stereo_setup['map1x'], stereo_setup['map1y'], cv2.INTER_LINEAR)

    disparity = stereo_setup['matcher'].compute(rect0, rect1).astype(np.float32) / 16.0
    valid_disp = disparity > 0

    _, Z = get_depth(img0, img1, stereo_setup, Z_MIN, Z_MAX)
    valid_Z = np.isfinite(Z)

    print(f"\n--- Frame {idx} (t={ts:.3f}) ---")
    if valid_disp.any():
        print(f"Disparity: valid={valid_disp.sum()}/{disparity.size} ({100 * valid_disp.mean():.1f}%), "
              f"range=[{disparity[valid_disp].min():.2f}, {disparity[valid_disp].max():.2f}], "
              f"mean={disparity[valid_disp].mean():.2f}")
    else:
        print("Disparity: NO valid pixels at all -- stereo matching found nothing")

    if valid_Z.any():
        print(f"Depth (post Z_MIN/Z_MAX clip): valid={valid_Z.sum()}/{Z.size} ({100 * valid_Z.mean():.1f}%), "
              f"range=[{np.nanmin(Z):.2f}, {np.nanmax(Z):.2f}] m, mean={np.nanmean(Z):.2f} m")
    else:
        print("Depth: NO valid pixels survive the Z_MIN/Z_MAX clip")

    sharpness0 = cv2.Laplacian(rect0, cv2.CV_64F).var()
    sharpness1 = cv2.Laplacian(rect1, cv2.CV_64F).var()
    print(f"Image sharpness (Laplacian variance): cam0={sharpness0:.1f}, cam1={sharpness1:.1f} "
          f"-- values below ~50 usually mean too little texture for stereo matching or flow")

    # Epipolar sanity check -- after correct rectification, a matching feature must sit
    # on the SAME row in both images. Draw horizontal guide lines to check this by eye.
    stacked = np.hstack([rect0, rect1])
    stacked_color = cv2.cvtColor(stacked, cv2.COLOR_GRAY2BGR)
    for row in range(0, stacked.shape[0], 40):
        cv2.line(stacked_color, (0, row), (stacked.shape[1], row), (0, 0, 255), 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(cv2.cvtColor(stacked_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Frame {idx}: rectified pair + epipolar lines")
    axes[0].axis('off')

    disp_show = disparity.copy()
    disp_show[~valid_disp] = np.nan
    im1 = axes[1].imshow(disp_show, cmap='jet')
    axes[1].set_title(f"Frame {idx}: disparity ({100 * valid_disp.mean():.0f}% valid)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.03)

    im2 = axes[2].imshow(Z, cmap='jet', vmin=Z_MIN, vmax=Z_MAX)
    axes[2].set_title(f"Frame {idx}: depth ({100 * valid_Z.mean():.0f}% valid)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.03)

    plt.tight_layout()
    out_path = f"stereo_diag_frame_{idx}.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")

print("\nDone. Check the epipolar-line images first: pick an object visible in both")
print("halves and confirm it sits on the same red guide line in cam0 and cam1. If it")
print("doesn't, rectification is broken and the depth values -- and therefore Vz and")
print("the whole VO scale -- can't be trusted no matter what the velocity solve does.")
print("Also check the valid-pixel percentages: underwater SGBM often has huge holes")
print("(<20-30% valid) if texture or the numDisparities/blockSize settings don't fit the scene.")