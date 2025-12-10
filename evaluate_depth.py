from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

# This speeds up evaluation 5x on some unix systems (OpenCV 3.3.1)
cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Stereo scale factor from original Monodepth
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1."""
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set."""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150.0

    # Device / CUDA
    use_cuda = (not opt.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pin_memory = device.type == "cuda"

    print("-> Evaluation device:", device)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        # -------------------------
        # Load model + prepare data
        # -------------------------
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)
        print("-> Loading weights from {}".format(opt.load_weights_folder))

        file_list_path = os.path.join(splits_dir, opt.eval_split, "test_files.txt")
        filenames = readlines(file_list_path)
        print("-> Using file list:", file_list_path)

        depth_model_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
        depth_model_dict = torch.load(depth_model_path, map_location="cpu")

        # Use SCAREDRAWDataset for EndoVis-style data
        dataset = datasets.SCAREDRAWDataset(
            opt.data_path,
            filenames,
            opt.height,
            opt.width,
            [0],
            4,
            is_train=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=opt.batch_size if hasattr(opt, "batch_size") else 16,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        # Instantiate DARES with the same LoRA configuration as training
        depth_model = networks.DARES(
            lora_mode=getattr(opt, "lora_mode", "dares"),
            lora_schedule_type=getattr(opt, "lora_schedule_type", "dares_front"),
            lora_base_rank=getattr(opt, "lora_base_rank", 14),
            lora_min_rank=getattr(opt, "lora_min_rank", 4),
            adalora_max_rank=getattr(opt, "adalora_max_rank", 16),
            adalora_total_rank_budget=getattr(opt, "adalora_total_rank_budget", 144),
        )

        model_dict = depth_model.state_dict()
        # Load only matching keys (LoRA + head)
        depth_model.load_state_dict({k: v for k, v in depth_model_dict.items() if k in model_dict}, strict=False)
        depth_model.to(device)
        depth_model.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(opt.width, opt.height))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].to(device)

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_model(input_color)
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps, axis=0)

    else:
        # -------------------------
        # Load precomputed disparities
        # -------------------------
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy")
            )
            pred_disps = pred_disps[eigen_to_benchmark_ids]

    # Optionally save raw disparities
    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split)
        )
        print("-> Saving predicted disparities to", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        return

    # KITTI benchmark special case (not used for EndoVis/SCARED)
    if opt.eval_split == "benchmark":
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        return

    # -------------------------
    # Load ground-truth depths
    # -------------------------
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    print("-> Loading GT depths from:", gt_path)
    gt_depths = np.load(gt_path, fix_imports=True, encoding="latin1")["data"]

    print("-> Evaluating on split '{}'".format(opt.eval_split))

    if opt.eval_stereo:
        print(
            "   Stereo evaluation - disabling median scaling, scaling by {}".format(
                STEREO_SCALE_FACTOR
            )
        )
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1.0 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array(
                [
                    0.40810811 * gt_height,
                    0.99189189 * gt_height,
                    0.03594771 * gt_width,
                    0.96405229 * gt_width,
                ]
            ).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]: crop[1], crop[2]: crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(
            " Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
                med, np.std(ratios / med)
            )
        )

    mean_errors = np.array(errors).mean(0)

    print(
        "\n  " + ("{:>8} | " * 7).format(
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
        )
    )
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
