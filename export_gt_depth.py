#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import cv2
from utils import readlines


def export_gt_depths_SCARED():
    parser = argparse.ArgumentParser(description="export_gt_depth")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)

    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)

    # Prefer test_files.txt
    test_list_path = os.path.join(split_folder, "test_files.txt")
    val_list_path = os.path.join(split_folder, "val_files.txt")

    if os.path.isfile(test_list_path):
        file_list_path = test_list_path
    elif os.path.isfile(val_list_path):
        file_list_path = val_list_path
    else:
        raise FileNotFoundError(f"No test or val files in {split_folder}")

    lines = readlines(file_list_path)
    print(f"Exporting ground truth depths for split '{opt.split}'")
    print("Using:", os.path.basename(file_list_path))

    gt_depths = []

    for i, line in enumerate(lines, start=1):
        folder, frame_id_str, _ = line.split()
        frame_id = int(frame_id_str)

        print(i)
        print(folder)

        # ---- FIXED PATH FOR YOUR DATASET ----
        # Your scene_points live here:
        # data/endovis/dataset_1/keyframe_1/data/scene_points/scene_pointsXXXXX.tiff
        f_str = f"scene_points{frame_id:06d}.tiff"
        gt_depth_path = os.path.join(
            opt.data_path,
            folder,
            "data",
            "scene_points",
            f_str
        )

        if not os.path.isfile(gt_depth_path):
            raise FileNotFoundError(f"GT file not found: {gt_depth_path}")

        depth_gt = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)

        # DARES uses the first channel (depth stored in channel 0)
        if depth_gt.ndim == 3:
            depth_gt = depth_gt[:, :, 0]

        # match DARES cropping behavior
        depth_gt = depth_gt[:1024, :]

        gt_depths.append(depth_gt.astype(np.float32))

    out_file = os.path.join(split_folder, "gt_depths.npz")
    print("Saving to:", out_file)
    np.savez_compressed(out_file, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_SCARED()
