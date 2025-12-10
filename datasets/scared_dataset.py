from __future__ import absolute_import, division, print_function

import os

import numpy as np
import PIL.Image as pil
import cv2
import skimage.transform  # kept for compatibility

from .mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        # Same intrinsics as original repo
        self.K = np.array([
            [0.82, 0.0, 0.5, 0.0],
            [0.0, 1.02, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Not actually used in our smoke test, but keep it
        self.full_res_shape = (1280, 1024)

        # Map side token -> stereo index (if ever needed)
        self.side_map = {"l": 2, "r": 3, "2": 2, "3": 3}

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        """
        Our smoke-test layout:

          data/endovis/
            dataset_1/
              keyframe_1/
                data/
                  frames/
                    0000000000.png

        Split lines look like:
          "dataset_1/keyframe_1 0000000000 2"

        mono_dataset.py passes:
          folder      = "dataset_1/keyframe_1 0000000000"
          frame_index = 0, 1, 2, ...

        So we:
          • strip the frame-id from `folder` (first token only)
          • always read from data/frames
        """
        base_folder = folder.split()[0]          # "dataset_1/keyframe_1"
        f_str = "{:010d}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(
            self.data_path,
            base_folder,
            "data",
            "frames",
            f_str,
        )
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        """
        Depth not used for Endovis-only training in our smoke test.
        Keep a stub that just raises if accidentally called.
        """
        raise RuntimeError("Depth loading not implemented for this smoke-test setup.")
