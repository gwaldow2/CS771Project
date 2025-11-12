# trainer_end_to_end.py
from __future__ import absolute_import, division, print_function

import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# project-local imports
import datasets  # keeps compatibility w/ SCARED
import networks
from networks import DARES

from utils import *
from layers import *

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Optional torchmetrics (MS-SSIM). We'll guard its use.
try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False

# Try explicit dataset class imports to avoid ambiguity
try:
    from datasets.scared_dataset import SCAREDRAWDataset
except Exception:
    # Fall back to attribute lookup on datasets package if needed
    SCAREDRAWDataset = getattr(datasets, "SCAREDRAWDataset")

try:
    from datasets.endoslam_dataset import EndoSLAMDataset
except Exception as e:
    raise ImportError("EndoSLAMDataset not found. Make sure datasets/endoslam_dataset.py exists.") from e


script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(script_path)


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # Ensure size multiples expected by the model
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # Device
        use_cuda = (not self.opt.no_cuda) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.pin_memory = self.device.type == "cuda"

        # Scales / frames
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        # If stereo is enabled for EndoVis/SCARED, include the "s" frame
        if self.opt.use_stereo and ("s" not in self.opt.frame_ids) and self.opt.dataset == "endovis":
            self.opt.frame_ids.append("s")

        # -----------------------
        # Models / Parameters
        # -----------------------
        self.models = {}
        self.parameters_to_train = []
        self.parameters_to_train_0 = []

        # Depth model (DARES)
        self.models["depth_model"] = DARES()
        self.models["depth_model"].to(self.device)
        # Only add trainable params (LoRA etc.)
        self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["depth_model"].parameters()))

        # AF-SfM Learner backbones/heads for position/pose/transform (appearance flow)
        # The repository README instructs placing weights in ./af_sfmlearner_weights/
        weights_dir = os.path.join(root_dir, "af_sfmlearner_weights")
        expected_files = [
            "position_encoder.pth",
            "position.pth",
            "transform_encoder.pth",
            "transform.pth",
            "pose_encoder.pth",
            "pose.pth",
        ]
        missing = [f for f in expected_files if not os.path.isfile(os.path.join(weights_dir, f))]
        if len(missing) > 0:
            raise FileNotFoundError(
                "Missing AF-SfMLearner weights in ./af_sfmlearner_weights. "
                "Please download and unpack as per README. Missing: {}".format(", ".join(missing))
            )

        # Position
        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2
        )
        self.models["position_encoder"].load_state_dict(torch.load(os.path.join(weights_dir, "position_encoder.pth"), map_location="cpu"))
        self.models["position_encoder"].to(self.device)
        self.parameters_to_train_0 += list(self.models["position_encoder"].parameters())

        self.models["position"] = networks.PositionDecoder(self.models["position_encoder"].num_ch_enc, self.opt.scales)
        self.models["position"].load_state_dict(torch.load(os.path.join(weights_dir, "position.pth"), map_location="cpu"))
        self.models["position"].to(self.device)
        self.parameters_to_train_0 += list(self.models["position"].parameters())

        # Transform (refinement network)
        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2
        )
        self.models["transform_encoder"].load_state_dict(torch.load(os.path.join(weights_dir, "transform_encoder.pth"), map_location="cpu"))
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())

        self.models["transform"] = networks.TransformDecoder(self.models["transform_encoder"].num_ch_enc, self.opt.scales)
        self.models["transform"].load_state_dict(torch.load(os.path.join(weights_dir, "transform.pth"), map_location="cpu"))
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

        # Pose (if enabled)
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                pose_encoder_path = os.path.join(weights_dir, "pose_encoder.pth")
                pose_decoder_path = os.path.join(weights_dir, "pose.pth")

                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=self.num_pose_frames
                )
                self.models["pose_encoder"].load_state_dict(torch.load(pose_encoder_path, map_location="cpu"))
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2
                )
                self.models["pose"].load_state_dict(torch.load(pose_decoder_path, map_location="cpu"))
            elif self.opt.pose_model_type == "shared":
                # If you ever switch to shared encoders, ensure self.models["encoder"] exists.
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames
                )
                self.models["pose"].load_state_dict(torch.load(os.path.join(weights_dir, "pose.pth"), map_location="cpu"))
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2
                )
                self.models["pose"].load_state_dict(torch.load(os.path.join(weights_dir, "pose.pth"), map_location="cpu"))

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        # Predictive mask (optional ablation)
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, (
                "When using predictive_mask, please disable automasking with --disable_automasking"
            )
            print('[INFO] predictive_mask enabled')
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales, num_output_channels=(len(self.opt.frame_ids) - 1)
            )
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        else:
            print('[INFO] predictive_mask disabled')

        # -----------------------
        # Optimizers / schedulers
        # -----------------------
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1
        )
        self.model_optimizer_0 = optim.Adam(self.parameters_to_train_0, 1e-4)
        self.model_lr_scheduler_0 = optim.lr_scheduler.StepLR(
            self.model_optimizer_0, self.opt.scheduler_step_size, 0.1
        )

        # Optional load of a previous run
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # -----------------------
        # Data
        # -----------------------
        datasets_dict = {
            "endovis": SCAREDRAWDataset,
            "endoslam": EndoSLAMDataset,
        }
        if self.opt.dataset not in datasets_dict:
            raise ValueError(f"Unknown dataset: {self.opt.dataset}. Choose from {list(datasets_dict.keys())}")

        self.dataset = datasets_dict[self.opt.dataset]

        # splits/<split>/{train,val}_files.txt
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        num_train_samples = len(train_filenames)
        self.num_total_steps = max(1, num_train_samples // max(1, self.opt.batch_size)) * max(1, self.opt.num_epochs)

        # Build datasets/loaders
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True
        )
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=self.pin_memory, drop_last=True
        )
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False
        )
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=min(1, self.opt.num_workers), pin_memory=self.pin_memory, drop_last=True
        )
        self.val_iter = iter(self.val_loader)

        # -----------------------
        # Logging / Metrics
        # -----------------------
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if (not self.opt.no_ssim) and _HAS_TORCHMETRICS:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        else:
            self.ms_ssim = None
            if not _HAS_TORCHMETRICS and not self.opt.no_ssim:
                print("[WARN] torchmetrics not found; falling back to L1 reprojection (set --no_ssim to silence)")

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width)).to(self.device)
        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width)).to(self.device)
        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width)).to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w).to(self.device)
            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w).to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"
        ]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset))
        )

        self.save_opts()

    # -----------------------
    # Modes
    # -----------------------
    def set_train_0(self):
        # Train only position networks
        for p in self.models["position_encoder"].parameters(): p.requires_grad = True
        for p in self.models["position"].parameters(): p.requires_grad = True

        for key in ["depth_model", "pose_encoder", "pose", "transform_encoder", "transform"]:
            if key in self.models:
                for p in self.models[key].parameters():
                    p.requires_grad = False

        self.models["position_encoder"].train()
        self.models["position"].train()

        for key in ["depth_model", "pose_encoder", "pose", "transform_encoder", "transform"]:
            if key in self.models:
                self.models[key].eval()

    def set_train(self):
        # Freeze position networks and train the rest
        for p in self.models["position_encoder"].parameters(): p.requires_grad = False
        for p in self.models["position"].parameters(): p.requires_grad = False

        for key in ["depth_model", "pose_encoder", "pose", "transform_encoder", "transform"]:
            if key in self.models:
                for p in self.models[key].parameters():
                    p.requires_grad = True

        self.models["position_encoder"].eval()
        self.models["position"].eval()

        for key in ["depth_model", "pose_encoder", "pose", "transform_encoder", "transform"]:
            if key in self.models:
                self.models[key].train()

    def set_eval(self):
        for key in ["depth_model", "transform_encoder", "transform", "pose_encoder", "pose"]:
            if key in self.models:
                self.models[key].eval()

    # -----------------------
    # Train / Epoch loops
    # -----------------------
    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            # Stage 0: position
            self.set_train_0()
            _, losses_0 = self.process_batch_0(inputs)
            self.model_optimizer_0.zero_grad()
            losses_0["loss"].backward()
            self.model_optimizer_0.step()

            # Stage 1: depth/pose/transform
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            phase = (batch_idx % max(1, self.opt.log_frequency) == 0)

            if phase:
                self.log_time(batch_idx, duration, losses["loss"].detach().cpu().data)
                self.log("train", inputs, outputs, losses)

            self.step += 1

        self.model_lr_scheduler.step()
        self.model_lr_scheduler_0.step()

    # -----------------------
    # Stage 0 (position)
    # -----------------------
    def process_batch_0(self, inputs):
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        outputs = {}
        outputs.update(self.predict_poses_0(inputs))
        losses = self.compute_losses_0(inputs, outputs)
        return outputs, losses

    def predict_poses_0(self, inputs):
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue

                inputs_all = [pose_feats[f_i], pose_feats[0]]
                inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                outputs_0 = self.models["position"](position_inputs)
                outputs_1 = self.models["position"](position_inputs_reverse)

                for scale in self.opt.scales:
                    outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                    outputs[("position", "high", scale, f_i)] = F.interpolate(
                        outputs[("position", scale, f_i)], [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=True
                    )
                    outputs[("registration", scale, f_i)] = self.spatial_transform(
                        inputs[("color", f_i, 0)], outputs[("position", "high", scale, f_i)]
                    )

                    outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                    outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                        outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=True
                    )
                    outputs[("occu_mask_backward", scale, f_i)], _ = self.get_occu_mask_backward(
                        outputs[("position_reverse", "high", scale, f_i)]
                    )
                    outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                        outputs[("position", "high", scale, f_i)],
                        outputs[("position_reverse", "high", scale, f_i)]
                    )

                transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                outputs_2 = self.models["transform"](transform_inputs)

                for scale in self.opt.scales:
                    outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                    outputs[("transform", "high", scale, f_i)] = F.interpolate(
                        outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=True
                    )
                    outputs[("refined", scale, f_i)] = (
                        outputs[("transform", "high", scale, f_i)]
                        * outputs[("occu_mask_backward", 0, f_i)].detach()
                        + inputs[("color", 0, 0)]
                    )
                    outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], 0.0, 1.0)

        return outputs

    def compute_losses_0(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0

            color = inputs[("color", 0, scale)]
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += get_smooth_loss(outputs[("position", scale, frame_id)], color)
                loss_registration += (
                    self.compute_reprojection_loss(outputs[("registration", scale, frame_id)],
                                                   outputs[("refined", scale, frame_id)].detach())
                    * occu_mask_backward
                ).sum() / (occu_mask_backward.sum() + 1e-7)

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)

            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    # -----------------------
    # Stage 1 (depth/pose/transform)
    # -----------------------
    def process_batch(self, inputs):
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        outputs = self.models["depth_model"](inputs["color_aug", 0, 0])

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, _disps_unused):
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue

                inputs_all = [pose_feats[f_i], pose_feats[0]]
                inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                outputs_0 = self.models["position"](position_inputs)
                outputs_1 = self.models["position"](position_inputs_reverse)

                for scale in self.opt.scales:
                    outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                    outputs[("position", "high", scale, f_i)] = F.interpolate(
                        outputs[("position", scale, f_i)], [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=True
                    )
                    outputs[("registration", scale, f_i)] = self.spatial_transform(
                        inputs[("color", f_i, 0)], outputs[("position", "high", scale, f_i)]
                    )

                    outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                    outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                        outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=True
                    )
                    outputs[("occu_mask_backward", scale, f_i)], outputs[("occu_map_backward", scale, f_i)] = \
                        self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                    outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                        outputs[("position", "high", scale, f_i)],
                        outputs[("position_reverse", "high", scale, f_i)]
                    )

                transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                outputs_2 = self.models["transform"](transform_inputs)

                for scale in self.opt.scales:
                    outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                    outputs[("transform", "high", scale, f_i)] = F.interpolate(
                        outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=True
                    )
                    outputs[("refined", scale, f_i)] = (
                        outputs[("transform", "high", scale, f_i)]
                        * outputs[("occu_mask_backward", 0, f_i)].detach()
                        + inputs[("color", 0, 0)]
                    )
                    outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], 0.0, 1.0)

                # Pose
                pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                axisangle, translation = self.models["pose"](pose_inputs)

                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0]
                )
        return outputs

    def generate_images_pred(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if not self.opt.v1_multiscale:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            source_scale = 0
            for _, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # PoseCNN special-casing
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / (depth + 1e-7)
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0
                    )

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)]
                )
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T
                )
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True
                )

                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                    cam_points, inputs[("K", source_scale)], T
                )

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if (self.ms_ssim is None) or self.opt.no_ssim:
            return l1_loss
        else:
            ms_ssim_loss = 1 - self.ms_ssim(pred, target)
            return 0.9 * ms_ssim_loss + 0.1 * l1_loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()

                loss_reprojection += (
                    self.compute_reprojection_loss(outputs[("color", frame_id, scale)],
                                                   outputs[("refined", scale, frame_id)])
                    * occu_mask_backward
                ).sum() / (occu_mask_backward.sum() + 1e-7)

                loss_transform += (
                    torch.abs(outputs[("refined", scale, frame_id)]
                              - outputs[("registration", 0, frame_id)].detach()).mean(1, True)
                    * occu_mask_backward
                ).sum() / (occu_mask_backward.sum() + 1e-7)

                loss_cvt += get_smooth_bright(
                    outputs[("transform", "high", scale, frame_id)],
                    inputs[("color", 0, 0)],
                    outputs[("registration", scale, frame_id)].detach(),
                    occu_mask_backward
                )

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += loss_reprojection / 2.0
            loss += self.opt.transform_constraint * (loss_transform / 2.0)
            loss += self.opt.transform_smoothness * (loss_cvt / 2.0)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    # -----------------------
    # Validation
    # -----------------------
    def val(self):
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch_val(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def process_batch_val(self, inputs):
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        # Here the original code had a branch for "shared" encoders;
        # we use the non-shared path (depth model consumes images directly).
        print('[INFO] validation step (non-shared pose model path)')
        features = self.models["depth_model"].backbone  # Not actually used; keep parity
        outputs = {}  # We call the same forward as in training for fair comparison
        outputs = self.models["depth_model"](inputs["color_aug", 0, 0])

        if self.opt.predictive_mask and "predictive_mask" in self.models:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses_val(inputs, outputs)
        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            registration_losses = []
            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                registration_losses.append(
                    ncc_loss(outputs[("registration", scale, frame_id)].mean(1, True),
                             target.mean(1, True))
                )

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, _ = torch.min(registration_losses, dim=1)
            loss += registration_losses.mean()
            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        # Following original sign convention
        losses["loss"] = -1 * total_loss
        return losses

    # -----------------------
    # Logging / IO
    # -----------------------
    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / max(duration, 1e-8)
        time_sofar = time.time() - self.start_time
        training_time_left = (
            (self.num_total_steps / max(self.step, 1) - 1.0) * time_sofar if self.step > 0 else 0
        )
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(
            self.epoch, batch_idx, samples_per_sec, float(loss),
            sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left))
        )

    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar(f"{l}", v, self.step)

        # Write up to four examples
        for j in range(min(4, self.opt.batch_size)):
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:
                    if frame_id == "s":
                        continue
                    writer.add_image(
                        f"brightness_{frame_id}_{s}/{j}",
                        outputs[("transform", "high", s, frame_id)][j].data, self.step
                    )
                    writer.add_image(
                        f"registration_{frame_id}_{s}/{j}",
                        outputs[("registration", s, frame_id)][j].data, self.step
                    )
                    writer.add_image(
                        f"refined_{frame_id}_{s}/{j}",
                        outputs[("refined", s, frame_id)][j].data, self.step
                    )
                    if s == 0:
                        writer.add_image(
                            f"occu_mask_backward_{frame_id}_{s}/{j}",
                            outputs[("occu_mask_backward", s, frame_id)][j].data, self.step
                        )
                writer.add_image(
                    f"disp_{s}/{j}",
                    normalize_image(outputs[("disp", s)][j]), self.step
                )

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}")
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        torch.save(self.model_optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            f"Cannot find folder {self.opt.load_weights_folder}"
        print(f"loading model from folder {self.opt.load_weights_folder}")

        for n in self.opt.models_to_load:
            print(f"Loading {n} weights...")
            path = os.path.join(self.opt.load_weights_folder, f"{n}.pth")
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location="cpu")
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        print("Adam is randomly initialized")


# -----------------------
# Standalone entry point
# -----------------------
if __name__ == "__main__":
    # Options are defined in options.py (MonodepthOptions)
    from options import MonodepthOptions
    opts = MonodepthOptions().parse()

    os.makedirs(opts.log_dir, exist_ok=True)

    trainer = Trainer(opts)
    trainer.train()
