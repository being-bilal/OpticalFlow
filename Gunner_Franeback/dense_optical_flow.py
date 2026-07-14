import cv2
import numpy as np

# Capturing the frame 1
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")
    
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Defines the Density of displayed arrows (distance between arrows)
STEP = 28

while True:
    # Capturing the frame 2
    ret, frame2 = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.2,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    vis = frame2.copy()
    h, w = gray.shape
    y, x = np.mgrid[STEP//2:h:STEP, STEP//2:w:STEP].astype(int)

    # Get flow vectors at sampled points
    fx = flow[y, x, 0]
    fy = flow[y, x, 1]

    # Draw arrows
    for (x0, y0, dx, dy) in zip(x.flatten(), y.flatten(), fx.flatten(), fy.flatten()):
        mag = np.sqrt(dx*dx + dy*dy)
        if mag < 1.0:
            continue
        end_point = (int(x0 + dx), int(y0 + dy))
        start_point = (int(x0), int(y0))

        # Draw arrow
        cv2.arrowedLine(
            vis,
            start_point,
            end_point,
            (0, 255, 0),
            2,
            tipLength=0.3
        )

        cv2.circle(vis, start_point, 1, (0, 255, 0), -1)

    cv2.imshow("Farneback Optical Flow (Arrows)", vis)

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()