# datasets/endoslam_dataset.py
from __future__ import annotations
import os
import re
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

_FRAME_RE = re.compile(r"^frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def _parse_cam_txt(cam_txt_path: str):
    kv = {}
    with open(cam_txt_path, "r") as f:
        for line in f:
            m = re.match(r"\s*([A-Za-z0-9_]+)\s*:\s*([-+Ee0-9\.]+)", line)
            if m:
                kv[m.group(1).lower()] = float(m.group(2))
    fx = kv.get("fx"); fy = kv.get("fy"); cx = kv.get("cx"); cy = kv.get("cy")
    if None in (fx, fy, cx, cy):
        raise ValueError(f"Missing fx/fy/cx/cy in {cam_txt_path}")
    return fx, fy, cx, cy


class EndoSLAMDataset(MonoDataset):
    """
    Split line (relative to --data_path), e.g.:
        Cameras/HighCam/Stomach-I/TumorfreeTrajectory_1/Frames  60
    """
    def __init__(self, *args, **kwargs):
        super(EndoSLAMDataset, self).__init__(*args, **kwargs)
        self.img_ext = ".jpg"
        self._raw_intrinsics = None  # fx, fy, cx, cy cached

    def _raw_K(self):
        if self._raw_intrinsics is None:
            cam_txt = os.path.join(
                self.data_path, "Cameras", "HighCam", "Calibration", "cam.txt.txt"
            )
            if not os.path.isfile(cam_txt):
                raise FileNotFoundError(f"Calibration file not found: {cam_txt}")
            self._raw_intrinsics = _parse_cam_txt(cam_txt)
        return self._raw_intrinsics  # (fx, fy, cx, cy)

    def _normalized_K_for_frames(self, frames_abs: str) -> np.ndarray:
        fx, fy, cx, cy = self._raw_K()
        probe = None
        for name in os.listdir(frames_abs):
            if _FRAME_RE.match(name):
                probe = os.path.join(frames_abs, name)
                break
        if probe is None:
            raise FileNotFoundError(f"No frame_* images in {frames_abs}")
        with pil.open(probe) as im:
            W, H = im.size
        fx_n, fy_n = fx / float(W), fy / float(H)
        cx_n, cy_n = cx / float(W), cy / float(H)
        K = np.array(
            [[fx_n, 0.0,  cx_n, 0.0],
             [0.0,  fy_n, cy_n, 0.0],
             [0.0,  0.0,  1.0,  0.0],
             [0.0,  0.0,  0.0,  1.0]], dtype=np.float32
        )
        return K

    def check_depth(self):
        return False

    def get_image_path(self, folder_rel: str, frame_index: int, side, *_):
        frames_dir = os.path.join(self.data_path, folder_rel)
        base = os.path.join(frames_dir, f"frame_{int(frame_index):06d}")
        for ext in (".jpg", ".png", ".jpeg"):
            p = base + ext
            if os.path.isfile(p):
                return p
        return base + self.img_ext

    def get_color(self, folder_rel: str, frame_index: int, side, do_flip: bool):
        img_path = self.get_image_path(folder_rel, frame_index, side)
        img = self.loader(img_path)
        if do_flip:
            img = img.transpose(pil.FLIP_LEFT_RIGHT)
        return img

    def __getitem__(self, index):
        # Use MonoDataset's robust parser so folder + index + side are consistent
        folder_rel, frame_index, side = self._parse_filename_line(self.filenames[index])

        # Compute intrinsics from the actual Frames directory
        frames_abs = os.path.join(self.data_path, folder_rel)
        self.K = self._normalized_K_for_frames(frames_abs)

        # Let MonoDataset.__getitem__ do the rest (it will parse the same line again
        # with _parse_filename_line to load images)
        return super().__getitem__(index)

