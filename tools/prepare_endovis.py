#!/usr/bin/env python3
import os
import cv2
import argparse
import random
from tqdm import tqdm

# This script lives in tools/, so build ENDOVIS_ROOT relative to here.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENDOVIS_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "endovis")

VIDEO_NAME = "rgb.mp4"
DEFAULT_SPLIT_NAME = "scared_smoke"


def extract_frames_from_video(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)

    existing = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
    if existing:
        print(f"[skip] Frames already present in {frames_dir} ({len(existing)} pngs)")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    print(f"[info] Extracting frames from {video_path} -> {frames_dir}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pbar = tqdm(total=total, ascii=True)

    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fname = f"{idx:010d}.png"
        cv2.imwrite(os.path.join(frames_dir, fname), frame)
        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"[done] Wrote {idx - 1} frames to {frames_dir}")


def find_sequences(endovis_root):
    """Return list of (rel_seq, video_path, frames_dir)."""
    seqs = []
    for root, dirs, files in os.walk(endovis_root):
        if VIDEO_NAME in files:
            video_path = os.path.join(root, VIDEO_NAME)
            keyframe_dir = os.path.dirname(video_path)          # .../data
            keyframe_root = os.path.dirname(keyframe_dir)       # .../keyframe_Y
            rel_seq = os.path.relpath(keyframe_root, endovis_root)  # dataset_X/keyframe_Y
            frames_dir = os.path.join(keyframe_dir, "frames")
            seqs.append((rel_seq, video_path, frames_dir))
    return seqs


def get_valid_neighbor_ids(frames_dir):
    """
    From a frames directory containing 0000000001.png etc., return
    a list of center frame IDs (strings) that have both neighbors:
    (i-1, i, i+1) all present.
    """
    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not frames:
        return []

    ids = [os.path.splitext(f)[0] for f in frames]
    width = len(ids[0])
    int_ids = [int(s) for s in ids]
    id_set = set(int_ids)

    centers = []
    for v in sorted(id_set):
        if (v - 1) in id_set and (v + 1) in id_set:
            centers.append(f"{v:0{width}d}")
    return centers


def write_split_single_sequence(target_seq, frames_dir, split_name, train_ratio, max_total, seed):
    """
    Write train/val split for a single Endovis sequence (target_seq),
    enforcing neighbor requirement and optional max_total.
    """
    ids = get_valid_neighbor_ids(frames_dir)
    if not ids:
        raise RuntimeError(f"No valid neighbor triplets found in {frames_dir}")

    if max_total > 0 and len(ids) > max_total:
        random.seed(seed)
        sampled = random.sample(ids, max_total)
        ids = sorted(sampled, key=lambda s: int(s))
        print(
            f"[info] Capping total Endovis samples from {len(get_valid_neighbor_ids(frames_dir))} "
            f"to {len(ids)} before train/val split."
        )

    cut = int(len(ids) * train_ratio)
    train_ids = ids[:cut]
    val_ids = ids[cut:]

    split_dir = os.path.join(SCRIPT_DIR, "..", "splits", split_name)
    os.makedirs(split_dir, exist_ok=True)
    train_path = os.path.join(split_dir, "train_files.txt")
    val_path = os.path.join(split_dir, "val_files.txt")

    print(f"[info] Writing split files in {split_dir}")
    with open(train_path, "w") as f:
        for fid in train_ids:
            f.write(f"{target_seq} {fid} 2\n")

    with open(val_path, "w") as f:
        for fid in val_ids:
            f.write(f"{target_seq} {fid} 2\n")

    print(f"[done] train_files.txt: {len(train_ids)} entries")
    print(f"[done] val_files.txt:   {len(val_ids)} entries")


def write_split_all_sequences(endovis_root, split_name, train_ratio, max_total, seed):
    """
    Aggregate valid neighbor centers from ALL sequences and write
    a global train/val split, respecting max_total.
    """
    seqs = find_sequences(endovis_root)
    if not seqs:
        raise RuntimeError(f"No {VIDEO_NAME} found under {endovis_root}")

    all_samples = []  # list of (rel_seq_norm, fid)
    for rel_seq, video_path, frames_dir in seqs:
        print(f"[info] Gathering centers for sequence: {rel_seq}")
        extract_frames_from_video(video_path, frames_dir)
        ids = get_valid_neighbor_ids(frames_dir)
        if not ids:
            print(f"[warn] No valid neighbor triplets in {frames_dir}, skipping.")
            continue

        rel_seq_norm = rel_seq.replace("\\", "/")
        for fid in ids:
            all_samples.append((rel_seq_norm, fid))

    if not all_samples:
        raise RuntimeError("No valid neighbor triplets found in any Endovis sequence.")

    print(f"[info] Total candidate centers across all sequences: {len(all_samples)}")

    random.seed(seed)
    if max_total > 0 and len(all_samples) > max_total:
        all_samples = random.sample(all_samples, max_total)
        print(f"[info] Capping total Endovis samples to {len(all_samples)}")

    # Shuffle before splitting so train/val mixes sequences
    random.shuffle(all_samples)

    cut = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:cut]
    val_samples = all_samples[cut:]

    split_dir = os.path.join(SCRIPT_DIR, "..", "splits", split_name)
    os.makedirs(split_dir, exist_ok=True)
    train_path = os.path.join(split_dir, "train_files.txt")
    val_path = os.path.join(split_dir, "val_files.txt")

    print(f"[info] Writing global split files in {split_dir}")
    with open(train_path, "w") as f:
        for rel_seq_norm, fid in train_samples:
            f.write(f"{rel_seq_norm} {fid} 2\n")

    with open(val_path, "w") as f:
        for rel_seq_norm, fid in val_samples:
            f.write(f"{rel_seq_norm} {fid} 2\n")

    print(f"[done] train_files.txt: {len(train_samples)} entries")
    print(f"[done] val_files.txt:   {len(val_samples)} entries")


def prepare_smoke(args):
    # Single-sequence "smoke" subset: dataset_1/keyframe_1
    target_rel = os.path.join("dataset_1", "keyframe_1")
    video_path = os.path.join(ENDOVIS_ROOT, target_rel, "data", VIDEO_NAME)
    frames_dir = os.path.join(ENDOVIS_ROOT, target_rel, "data", "frames")

    if not os.path.isfile(video_path):
        raise RuntimeError(f"rgb.mp4 not found at {video_path}")

    extract_frames_from_video(video_path, frames_dir)
    target_rel_norm = target_rel.replace("\\", "/")
    write_split_single_sequence(
        target_rel_norm,
        frames_dir,
        args.split_name,
        args.train_ratio,
        args.max_total,
        args.seed,
    )

    print(
        "\n[OK] Endovis smoke split ready. Run e.g.:\n"
        "python train_end_to_end.py "
        f"--dataset endovis --data_path data/endovis --split {args.split_name} "
        "--model_name smoke_cpu --no_cuda --num_epochs 1 --batch_size 1 "
        "--num_workers 0 --height 384 --width 384"
    )


def prepare_all_with_split(args):
    """
    New non-smoke behavior:
    - Extract frames for all sequences (if not already extracted)
    - Build a global neighbor-checked pool
    - Apply max_total, train_ratio
    - Write train/val split.
    """
    write_split_all_sequences(
        ENDOVIS_ROOT,
        args.split_name,
        args.train_ratio,
        args.max_total,
        args.seed,
    )

    print(
        "\n[OK] Endovis global split ready. Run e.g.:\n"
        "python train_end_to_end.py "
        f"--dataset endovis --data_path data/endovis --split {args.split_name} "
        "--model_name endovis_global --no_cuda --num_epochs 1 --batch_size 1 "
        "--num_workers 0 --height 384 --width 384"
    )


def main():
    if not os.path.isdir(ENDOVIS_ROOT):
        raise RuntimeError(f"Endovis root not found: {ENDOVIS_ROOT}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only prepare dataset_1/keyframe_1 and write a split (default scared_smoke).",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=DEFAULT_SPLIT_NAME,
        help="Name of the Endovis split directory (default: scared_smoke).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of samples for train (rest go to val).",
    )
    parser.add_argument(
        "--max_total",
        type=int,
        default=0,
        help="Optional cap on total number of Endovis samples (train+val). 0 = no cap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for Endovis subsampling when using --max_total.",
    )
    args = parser.parse_args()

    if args.smoke:
        prepare_smoke(args)
    else:
        prepare_all_with_split(args)


if __name__ == "__main__":
    main()
