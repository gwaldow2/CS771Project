#!/usr/bin/env python3
"""
Build EndoSLAM splits for DARES with your existing folder structure.

- Walks data_root (e.g., ./data/endoSLAM or ./data/endoSLAM/Cameras)
- Finds every ".../Frames/" folder
- Collects files named "frame_######.jpg|jpeg|png"
- Writes lines like:
    Cameras/HighCam/Stomach-III/TumorfreeTrajectory_1/Frames|208

IMPORTANT:
    This script only writes frame indices that have temporal neighbors for
    monodepth-style training, i.e. we require that (idx-1) and (idx+1) exist
    in the same Frames directory. This matches the use of frame_idxs = [0,-1,1]
    in MonoDataset.__getitem__.
"""

import argparse, os, re, sys, random

FRAME_DIR_NAME = "Frames"
PATTERN = re.compile(r"^frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

# Offsets expected by the loader around the "center" frame index.
# For MonodepthOptions default frame_ids = [0, -1, 1], this should be [-1, 1].
NEIGHBOR_OFFSETS = (-1, 1)


def collect_lines(data_root: str):
    lines = []

    for dirpath, dirnames, filenames in os.walk(data_root):
        if os.path.basename(dirpath).lower() != FRAME_DIR_NAME.lower():
            continue

        # Collect all integer indices present in this Frames directory
        idxs = set()
        for name in filenames:
            m = PATTERN.match(name)
            if not m:
                continue
            idx = int(m.group(1))
            idxs.add(idx)

        if not idxs:
            continue

        idxs = sorted(idxs)
        rel_folder = os.path.relpath(dirpath, data_root).replace("\\", "/")

        # For each potential "center" frame, require that all neighbor indices exist
        for idx in idxs:
            ok = True
            for off in NEIGHBOR_OFFSETS:
                if (idx + off) not in idxs:
                    ok = False
                    break
            if not ok:
                continue

            # Use pipe to separate folder and frame index so spaces in folder are harmless
            lines.append(f"{rel_folder}|{idx}")

    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data/endoSLAM",
                    help="Path to EndoSLAM root (e.g., ./data/endoSLAM).")
    ap.add_argument("--out_dir", default="./splits/endoslam",
                    help="Where to write the split files.")
    ap.add_argument("--train_ratio", type=float, default=0.95)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before splitting.")
    args = ap.parse_args()

    data_root = os.path.normpath(args.data_root)
    lines = collect_lines(data_root)
    if not lines:
        print(f"No valid frame centers found under {data_root}. "
              f"Expect .../Frames/frame_XXXXXX.jpg and contiguous neighbors.",
              file=sys.stderr)
        sys.exit(1)

    if args.shuffle:
        random.seed(1234)
        random.shuffle(lines)

    n_train = max(1, int(len(lines) * args.train_ratio))
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train_files.txt"), "w", newline="\n") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(args.out_dir, "val_files.txt"), "w", newline="\n") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"Wrote {len(train_lines)} train and {len(val_lines)} val lines to {args.out_dir}")
    if train_lines:
        print("Example:", train_lines[0])


if __name__ == "__main__":
    main()
