import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch

from raft import RAFT
from utils.utils import InputPadder


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


def frame_to_tensor(frame, max_side=640):
    # cv2 gives BGR, RAFT expects RGB
    h, w = frame.shape[:2]
    scale = max_side / max(w, h)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def draw_flow_arrows(img, flow, step=16, scale=1.0, min_mag=0.5):
    """
    img:  HxWx3 uint8 RGB
    flow: HxWx2 float (dx, dy) per pixel
    step: grid spacing in pixels between sampled arrows
    scale: multiply displacement for visibility (flow vectors are often small)
    min_mag: skip drawing arrows below this magnitude (reduces clutter from noise)
    """
    h, w = img.shape[:2]
    vis = img.copy()

    y_coords = np.arange(step // 2, h, step)
    x_coords = np.arange(step // 2, w, step)

    for y in y_coords:
        for x in x_coords:
            dx, dy = flow[y, x]
            mag = np.hypot(dx, dy)
            if mag < min_mag:
                continue
            x2 = int(round(x + dx * scale))
            y2 = int(round(y + dy * scale))
            cv2.arrowedLine(
                vis, (x, y), (x2, y2),
                color=(0, 255, 0), thickness=1,
                tipLength=0.35
            )
    return vis


def viz(img, flo, writer=None, step=24, scale=1.0, min_mag=0.5):
    img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    arrow_img = draw_flow_arrows(img, flo, step=step, scale=scale, min_mag=min_mag)
    out_bgr = arrow_img[:, :, [2, 1, 0]]  # RGB -> BGR for display/write

    cv2.imshow('flow arrows', out_bgr)
    if writer is not None:
        writer.write(out_bgr)

    # non-blocking; 'q' to quit early
    return cv2.waitKey(1) & 0xFF == ord('q')


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    model = model.module
    model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    writer = None

    with torch.no_grad():
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("Video has no frames")

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            image1 = frame_to_tensor(prev_frame, max_side=args.max_side)
            image2 = frame_to_tensor(curr_frame, max_side=args.max_side)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # lazily create writer once we know the actual (post-resize) frame size
            if args.save and writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                h, w = image1.shape[2], image1.shape[3]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            if viz(image1, flow_up, writer, step=args.step, scale=args.arrow_scale, min_mag=args.min_mag):
                break

            prev_frame = curr_frame

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--video', help="path to input video file")
    parser.add_argument('--save', default=None, help="optional path to save output video")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--max_side', type=int, default=640, help='resize longer side to this before inference')
    parser.add_argument('--step', type=int, default=24, help='pixel spacing between arrows')
    parser.add_argument('--arrow_scale', type=float, default=1.0, help='multiply displacement for visibility')
    parser.add_argument('--min_mag', type=float, default=0.5, help='hide arrows below this magnitude (denoise)')
    args = parser.parse_args()

    demo(args)