# datasets/mono_dataset.py
from __future__ import absolute_import, division, print_function

import os
import random
import re
import numpy as np
from PIL import Image
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pillow >=10 uses Image.Resampling; fall back otherwise
try:
    _RESAMPLE = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except Exception:
    _RESAMPLE = Image.LANCZOS


def pil_loader(path):
    """Open path as file to avoid ResourceWarning (Pillow quirk)."""
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class MonoDataset(data.Dataset):
    """
    Superclass for monocular dataloaders.

    Robust split line parsing supported:

      • Pipe-delimited (preferred):
          "<relative_folder>|<frame_idx>|<side?>"
        Examples:
          "Cameras/HighCam/HighCam/Small Intestine/TumorfreeTrajectory_1|60"
          "dataset1/sceneA|120|l"

      • Whitespace-delimited (compat mode), where <relative_folder> may contain spaces.
        The parser reads optional trailing <side> and <frame_idx>, everything else is folder.
        Examples:
          "Cameras/HighCam/HighCam/Small Intestine/TumorfreeTrajectory_1 60"
          "dataset1/sceneA 120 l"

    Notes
    -----
    * 'relative_folder' is joined under --data_path by subclasses.
    * 'side' is only meaningful when stereo frames ("s") are requested.
    * Subclasses must define: self.K (intrinsics), get_color(), check_depth(), get_depth().
    """

    def __init__(
        self,
        data_path,
        filenames,
        height,
        width,
        frame_idxs,
        num_scales,
        is_train=False,
        img_ext=".png",
    ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        self.interp = _RESAMPLE

        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext  # subclasses may override per sample

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # Color jitter ranges (newer torchvision expects tuples)
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # construct once to validate argument style
            _ = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        except TypeError:
            # very old torchvision fallback (scalar deltas)
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # Multi-scale resize ops
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp
            )

        # Let subclass decide if depth GT is available
        self.load_depth = self.check_depth()

    # ---- Subclass hooks -------------------------------------------------------

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    # ---- Helpers --------------------------------------------------------------

    def _parse_filename_line(self, raw_line):
        """
        Parse a line from *_files.txt into (folder, frame_index, side).

        Supports:
          - 'folder|frame|side?'
          - 'folder ... possibly with spaces ... [frame] [side]'

        Returns
        -------
        folder : str           (relative to self.data_path, NOT normalized here)
        frame_index : int
        side : Optional[str]   ('l'/'r'/'s' or None)
        """
        line = raw_line.strip()

        # Preferred: pipe-delimited
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 1 or parts[0] == "":
                raise ValueError(f"Bad split line (empty folder): {raw_line}")
            folder = parts[0]
            frame_index = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
            side = parts[2] if len(parts) > 2 and parts[2] in {"l", "r", "s"} else None
            return folder, frame_index, side

        # Fallback: whitespace tokens, but folder may include spaces.
        toks = line.split()
        if not toks:
            raise ValueError(f"Bad split line (empty): {raw_line}")

        side = None
        frame_index = 0

        # Optional side token at end
        if toks and toks[-1] in {"l", "r", "s"}:
            side = toks[-1]
            toks = toks[:-1]

        # Optional integer frame index at end
        if toks and re.fullmatch(r"-?\d+", toks[-1]):
            frame_index = int(toks[-1])
            toks = toks[:-1]

        if not toks:
            raise ValueError(f"Bad split line (missing folder): {raw_line}")

        folder = " ".join(toks)
        return folder, frame_index, side

    def preprocess(self, inputs, color_aug):
        """
        Resize colour images to required scales and apply consistent augmentation.
        We apply the SAME color augmentation to all images in the item.
        """
        # Build scaled image pyramid from native (-1) to scales 0..N-1
        base_keys = [k for k in list(inputs) if isinstance(k, tuple) and k[0] == "color" and k[2] == -1]
        for (n, im, _) in base_keys:
            # create scale 0..num_scales-1 from native (-1)
            for s in range(self.num_scales):
                src_key = (n, im, -1) if s == 0 else (n, im, s - 1)
                dst_key = (n, im, s)
                inputs[dst_key] = self.resize[s](inputs[src_key])

        # To tensor + aug
        color_keys = [k for k in list(inputs) if isinstance(k, tuple) and k[0] == "color" and k[2] >= 0]
        for (n, im, s) in color_keys:
            img = inputs[(n, im, s)]
            inputs[(n, im, s)] = self.to_tensor(img)
            inputs[(n + "_aug", im, s)] = self.to_tensor(color_aug(img))

    # ---- Dataset API ----------------------------------------------------------

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Returns one training item.

        Keys:
            ("color", <frame_id>, <scale>)
            ("color_aug", <frame_id>, <scale>)
            ("K", scale), ("inv_K", scale)
            "stereo_T" (if stereo requested)
            "depth_gt" (if available)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # --- Robust line parsing (handles spaces & pipes) ---
        folder, frame_index, side = self._parse_filename_line(self.filenames[index])

        # --- Load images for requested frames (temporal or stereo) ---
        for i in self.frame_idxs:
            if i == "s":
                # opposite eye for stereo; requires 'side'
                if side is None:
                    # If user requested stereo frames but split has no side, be explicit.
                    raise ValueError(
                        "Stereo frame requested (i=='s') but 'side' token missing in split line:\n"
                        f"  {self.filenames[index]}"
                    )
                other_side = {"r": "l", "l": "r"}.get(side)
                if other_side is None:
                    raise ValueError(f"Unrecognized side '{side}' in split line: {self.filenames[index]}")
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # --- Intrinsics per scale (expects self.K normalized by width/height) ---
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # --- Color augmentation (consistent across frames) ---
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            color_aug = (lambda x: x)

        # Build pyramid + tensors + color_aug
        self.preprocess(inputs, color_aug)

        # Drop native-res images to save memory
        for i in self.frame_idxs:
            if ("color", i, -1) in inputs:
                del inputs[("color", i, -1)]
            if ("color_aug", i, -1) in inputs:
                del inputs[("color_aug", i, -1)]

        # Optional depth supervision
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            depth_gt = np.expand_dims(depth_gt, 0).astype(np.float32)
            inputs["depth_gt"] = torch.from_numpy(depth_gt)

        # Optional fixed stereo baseline if stereo is requested AND side is known
        if ("s" in self.frame_idxs) and (side is not None):
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs
