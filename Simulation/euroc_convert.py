#!/usr/bin/env python3
"""euroc_convert.py — repackage a sauvc_traj_recorder trajectory folder into the EuRoC/ASL
layout that ros2_orb_slam3's mono_driver_node.py expects.

WHY THIS IS NEEDED
-------------------
trajectory_recorder_node.py writes images as camera_front/000000_<t>.png plus a separate
camera_front_index.csv (idx, sec, nanosec, t, filename). mono_driver_node.py's
get_image_dataset_asl() instead expects a bare directory of images named
<timestamp_ns>.png — it derives the per-frame ROS timestep by doing
float(filename.split('.')[0]), i.e. the ENTIRE filename stem must be one integer
nanosecond timestamp (exactly like the EuRoC dataset: 1403638538577829376.png).

This script copies (not moves) images from the recorder's output into that layout using
the sec/nanosec columns already in the index CSV (exact integers — safer than
reparsing the rounded 't' float column). It does not touch your original recording.

USAGE
    python3 euroc_convert.py /path/to/trajectory_03 my_seq_name \\
        --dataset-root ~/Robotics_Job/sauvc_ws/src/ros2_orb_slam3/TEST_DATASET \\
        --cam camera_front

This produces:
    TEST_DATASET/my_seq_name/mav0/cam0/data/<t_ns>.png
    TEST_DATASET/my_seq_name/mav0/cam0/data.csv        (EuRoC-style index, not required by
                                                          mono_driver_node.py but standard)

Then run:
    ros2 run ros2_orb_slam3 mono_driver_node.py --ros-args \\
        -p settings_name:=<your_settings_yaml_stem> -p image_seq:=my_seq_name
"""

import argparse
import csv
import os
import shutil


def convert(traj_dir, seq_name, dataset_root, cam):
    index_csv = os.path.join(traj_dir, f'{cam}_index.csv')
    src_dir = os.path.join(traj_dir, cam)
    if not os.path.isfile(index_csv):
        raise SystemExit(f'{index_csv} not found — did you record {cam}_topic in this take?')

    out_dir = os.path.join(dataset_root, seq_name, 'mav0', 'cam0', 'data')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with open(index_csv, newline='') as f:
        for r in csv.DictReader(f):
            t_ns = int(r['sec']) * 1_000_000_000 + int(r['nanosec'])
            rows.append((t_ns, r['filename']))
    rows.sort(key=lambda row: row[0])

    data_csv_path = os.path.join(dataset_root, seq_name, 'mav0', 'cam0', 'data.csv')
    with open(data_csv_path, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['#timestamp [ns]', 'filename'])
        for t_ns, filename in rows:
            src = os.path.join(src_dir, filename)
            if not os.path.isfile(src):
                print(f'  WARNING: {src} listed in index but missing on disk, skipping')
                continue
            dst = os.path.join(out_dir, f'{t_ns}.png')
            shutil.copy2(src, dst)
            writer.writerow([t_ns, f'{t_ns}.png'])

    print(f'wrote {len(rows)} frames -> {out_dir}')
    print(f'wrote index -> {data_csv_path}')
    print(f'\nrun with: -p image_seq:={seq_name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('traj_dir', help='sauvc_traj_recorder trajectory folder, e.g. trajectory_03')
    ap.add_argument('seq_name', help='name for this sequence under ros2_orb_slam3/TEST_DATASET')
    ap.add_argument('--dataset-root', required=True,
                    help='path to ros2_orb_slam3/TEST_DATASET on this machine')
    ap.add_argument('--cam', default='camera_front', choices=['camera_front', 'camera_down'],
                    help='which recorded camera stream to convert (default: camera_front)')
    args = ap.parse_args()

    dataset_root = os.path.expanduser(args.dataset_root)
    convert(os.path.expanduser(args.traj_dir), args.seq_name, dataset_root, args.cam)


if __name__ == '__main__':
    main()