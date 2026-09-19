import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils.utils import InputPadder


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


def load_image(imfile, max_side=640):
    img = Image.open(imfile).convert('RGB')
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def draw_flow_arrows(img, flow, step=16, scale=1.0, min_mag=0.5):
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


def viz(img, flo, step=16, scale=1.0, min_mag=0.5):
    img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    arrow_img = draw_flow_arrows(img, flo, step=step, scale=scale, min_mag=min_mag)

    cv2.imshow('flow arrows', arrow_img[:, :, [2, 1, 0]])  # RGB -> BGR for display
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, step=args.step, scale=args.arrow_scale, min_mag=args.min_mag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--step', type=int, default=16, help='pixel spacing between arrows')
    parser.add_argument('--arrow_scale', type=float, default=1.0, help='multiply displacement for visibility')
    parser.add_argument('--min_mag', type=float, default=0.5, help='hide arrows below this magnitude (denoise)')
    args = parser.parse_args()

    demo(args)