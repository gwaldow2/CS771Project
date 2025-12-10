#!/usr/bin/env python3
"""
Build EndoSLAM splits for DARES with your existing folder structure.

- Walks data_root (e.g., ./data/endoslam)
- Finds every ".../Frames/" folder
- Collects files named "frame_######.jpg|jpeg|png"
- For each candidate center index i, only keeps it if *all* required
  neighbor indices (e.g. i-1, i, i+1) exist in that folder.
- Optionally caps the total number of (train+val) samples with --max_total.
- Writes lines like:
    Cameras/HighCam/HighCam/Stomach-I/TumorfreeTrajectory_1/Frames  60
  into ./splits/endoslam/{train_files.txt,val_files.txt}
"""

import argparse
import os
import re
import sys
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FRAME_DIR_NAME = "Frames"
PATTERN = re.compile(r"^frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

# Neighbor offsets required around each "center" frame index.
# For standard Monodepth-style training with frame_ids [-1, 0, 1],
# we need {-1, 0, +1}.
NEIGHBOR_OFFSETS = (-1, 0, 1)


def collect_lines(data_root: str):
    lines = []
    total_centers = 0
    dropped = 0

    for dirpath, dirnames, filenames in os.walk(data_root):
        if os.path.basename(dirpath).lower() != FRAME_DIR_NAME.lower():
            continue

        # Gather all frame indices in this Frames directory
        idxs = []
        for name in filenames:
            m = PATTERN.match(name)
            if m:
                idxs.append(int(m.group(1)))

        if not idxs:
            continue

        idxs_set = set(idxs)

        # For each candidate center index, require all neighbors exist
        for idx in sorted(idxs):
            total_centers += 1
            ok = True
            for off in NEIGHBOR_OFFSETS:
                if (idx + off) not in idxs_set:
                    ok = False
                    break
            if not ok:
                dropped += 1
                continue

            rel_folder = os.path.relpath(dirpath, data_root).replace("\\", "/")
            lines.append(f"{rel_folder} {idx}")

    print(
        f"[collect_lines] Found {total_centers} candidate centers, "
        f"kept {len(lines)} with full neighbors, dropped {dropped}."
    )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        default="./data/endoslam",
        help="Path to EndoSLAM root (e.g., ./data/endoslam). "
             "If relative, it is resolved w.r.t. the repo root (tools/..).",
    )
    ap.add_argument(
        "--out_dir",
        default="./splits/endoslam",
        help="Where to write the split files (relative to repo root by default).",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Fraction of samples for train (rest go to val).",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before splitting (recommended when using --max_total).",
    )
    ap.add_argument(
        "--max_total",
        type=int,
        default=0,
        help="Optional cap on total number of samples (train+val). 0 = no cap.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for shuffling/subsampling.",
    )
    args = ap.parse_args()

    # Resolve data_root relative to repo root (tools/..)
    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(SCRIPT_DIR, "..", data_root))

    print(f"[info] Using EndoSLAM root: {data_root}")
    if not os.path.isdir(data_root):
        print(f"[error] data_root does not exist or is not a directory: {data_root}", file=sys.stderr)
        sys.exit(1)

    lines = collect_lines(data_root)
    if not lines:
        print(
            f"No valid centers found under {data_root}. "
            f"Expect .../Frames/frame_XXXXXX.jpg with neighbors.",
            file=sys.stderr)
        sys.exit(1)

    # Resolve out_dir relative to repo root as well
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "..", out_dir))

    # Shuffle if requested or if we are going to subsample
    if args.shuffle or (args.max_total > 0 and len(lines) > args.max_total):
        random.seed(args.seed)
        random.shuffle(lines)

    # Optional cap on total number of samples
    if args.max_total > 0 and len(lines) > args.max_total:
        print(
            f"[info] Capping total samples from {len(lines)} to {args.max_total} "
            f"before train/val split."
        )
        lines = lines[:args.max_total]

    n_train = max(1, int(len(lines) * args.train_ratio))
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    os.makedirs(out_dir, exist_ok=True)
    train_file = os.path.join(out_dir, "train_files.txt")
    val_file = os.path.join(out_dir, "val_files.txt")

    with open(train_file, "w", newline="\n") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(val_file, "w", newline="\n") as f:
        f.write("\n".join(val_lines) + "\n")

    print(
        f"Wrote {len(train_lines)} train and {len(val_lines)} val lines to {out_dir}"
    )
    if train_lines:
        print("Example:", train_lines[0])


if __name__ == "__main__":
    main()
