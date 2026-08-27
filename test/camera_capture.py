"""
Capture RGB checkerboard images from the RealSense D435i for camera calibration.

Controls (with the preview window focused):
  SPACE  - save current frame
  q      - quit

Saved images go to ./calib_images/img_00.png, img_01.png, ...
"""

import os
import cv2
import numpy as np
import pyrealsense2 as rs

OUTPUT_DIR = "calib_images"
WIDTH, HEIGHT, FPS = 1280, 720, 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
pipeline.start(config)

print("Live preview started.")
print("Press SPACE to save a frame, q to quit.")

count = 0
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())

        preview = image.copy()
        cv2.putText(
            preview, f"Saved: {count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        cv2.imshow("RealSense RGB - SPACE to save, q to quit", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            filename = os.path.join(OUTPUT_DIR, f"img_{count:02d}.png")
            cv2.imwrite(filename, image)
            print(f"Saved {filename}")
            count += 1
        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print(f"Done. {count} images saved to '{OUTPUT_DIR}/'.")