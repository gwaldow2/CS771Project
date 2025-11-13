#!/usr/bin/env python3
"""
Build EndoSLAM splits for DARES with your existing folder structure.

- Walks data_root (e.g., ./data/endoSLAM or ./data/endoSLAM/Cameras)
- Finds every ".../Frames/" folder
- Collects files named "frame_######.jpg|jpeg|png"
- Writes lines like:
    Cameras/HighCam/HighCam/Stomach-I/TumorfreeTrajectory_1/Frames  60
  into ./splits/endoslam/{train_files.txt,val_files.txt}
"""

import argparse, os, re, sys, random

FRAME_DIR_NAME = "Frames"
PATTERN = re.compile(r"^frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def collect_lines(data_root: str):
    lines = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        if os.path.basename(dirpath).lower() != FRAME_DIR_NAME.lower():
            continue
        for name in sorted(filenames):
            m = PATTERN.match(name)
            if not m:
                continue
            idx = int(m.group(1))
            rel_folder = os.path.relpath(dirpath, data_root).replace("\\", "/")
            # Use a pipe to separate folder and frame index so spaces are irrelevant
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
        print(f"No frames found under {data_root}. "
              f"Expect .../Frames/frame_XXXXXX.jpg", file=sys.stderr)
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
