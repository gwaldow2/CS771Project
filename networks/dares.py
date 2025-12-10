from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DepthAnythingForDepthEstimation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import math
from torch.nn.parameter import Parameter


# MODIFICATION: LoRA Schedule/AdaLoRA -----
class _LoRA_qkv(nn.Module):
    """Wrapper around a single Linear (q/k/v) with optional LoRA and dynamic rank gating."""

    def __init__(
        self,
        w: nn.Module,
        linear_a: nn.Module,
        linear_b: nn.Module,
        active_rank: int | None = None,
    ):
        super().__init__()
        self.w = w
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dim = w.in_features
        # active_rank <= out_features of linear_a; if None, use full rank
        self.active_rank = active_rank

    def forward(self, x):
        # Base projection
        W = self.w(x)

        if self.linear_a is None or self.linear_b is None:
            return W

        a_out = self.linear_a(x)  # [B, N, r_max] or [B, N, r]

        # If active_rank is set and smaller than allocated rank, gate to the first active_rank dims.
        if (
            self.active_rank is not None
            and self.active_rank > 0
            and self.active_rank < a_out.shape[-1]
        ):
            r = self.active_rank
            a_active = a_out[..., :r]  # [B, N, r]

            # linear_b: nn.Linear(alloc_rank, dim)
            # weight shape: [dim, alloc_rank]
            # We want only the first r input channels -> slice columns, not rows.
            weight = self.linear_b.weight[:, :r]  # [dim, r]
            deltaW = torch.matmul(a_active, weight.t())  # [B, N, dim]

            if self.linear_b.bias is not None:
                deltaW = deltaW + self.linear_b.bias
        else:
            # No gating or full rank: just use the standard linear
            deltaW = self.linear_b(a_out)

        return W + deltaW


# ---------------------
## Commented out code which it replaces
# class _LoRA_qkv(nn.Module):
#     """In Dinov2 it is implemented as
#     self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#     B, N, C = x.shape
#     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv.unbind(0)
#     """
#
#     def __init__(
#             self,
#             w: nn.Module,
#             linear_a: nn.Module,
#             linear_b: nn.Module
#     ):
#         super().__init__()
#         self.w = w
#         self.linear_a = linear_a
#         self.linear_b = linear_b
#         self.dim = w.in_features
#
#     def forward(self, x):
#         W = self.w(x)
#         residual = W.clone()
#         deltaW = self.linear_b(self.linear_a(x))
#
#         W += deltaW
#         return W
# -------------------------------


class DepthAnythingDepthEstimationHead(nn.Module):
    def __init__(self, model_head):
        super().__init__()
        self.conv1 = model_head.conv1
        self.conv2 = model_head.conv2
        self.activation1 = nn.ReLU()
        self.conv3 = model_head.conv3
        self.activation2 = nn.Sigmoid()

    def forward(self, hidden_states, height, width):
        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (int(height), int(width)),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        return predicted_depth


# MODIFICATION: LoRA Schedule/AdaLoRA -----
def _build_rank_vector(num_layers: int, base_rank: int, min_rank: int, schedule_type: str):
    """
    Build a per-layer rank vector given a schedule type.

    - 'dares_front': front-heavy, decaying ranks like the original DARES (14,14,12,12,...).
    - 'uniform':     same rank for all layers.
    - 'back_heavy':  mirror of 'dares_front' (larger ranks in deeper layers).
    - 'u_shape':     larger ranks at shallow & deep layers, smaller in the middle.
    """
    min_rank = max(1, min_rank)
    base_rank = max(min_rank, base_rank)

    if num_layers <= 0:
        return []

    if schedule_type == "uniform":
        return [base_rank] * num_layers

    # Front-heavy decay similar to the DARES vector-LoRA pattern
    if schedule_type == "dares_front":
        ranks = []
        current = base_rank
        for i in range(num_layers):
            if i > 0 and i % 2 == 0 and current > min_rank:
                current = max(min_rank, current - 2)
            ranks.append(current)
        return ranks

    # Back-heavy: reverse of dares_front
    if schedule_type == "back_heavy":
        front = _build_rank_vector(num_layers, base_rank, min_rank, "dares_front")
        return front[::-1]

    # U-shaped: max of front-heavy and back-heavy to lift both ends
    if schedule_type == "u_shape":
        front = _build_rank_vector(num_layers, base_rank, min_rank, "dares_front")
        back = front[::-1]
        return [max(f, b) for f, b in zip(front, back)]

    # Fallback (shouldn't happen): uniform
    return [base_rank] * num_layers


# ---------------------
## Commented out code which it replaces
# (no previous helper; this is a new utility for rank scheduling)
# -------------------------------


# MODIFICATION: LoRA Schedule/AdaLoRA -----
class LoRAInitializer:
    """
    Helper that:
      * Freezes the DepthAnything backbone
      * Injects LoRA adapters into attention q/k/v
      * Supports:
          - Original DARES vector-LoRA (front-heavy ranks)
          - Static depth-dependent rank schedules
          - AdaLoRA-style max-rank with per-layer active_rank gating
    """

    def __init__(
        self,
        model,
        mode: str = "dares",               # 'dares', 'schedule', 'adalora'
        r=None,                            # optional explicit rank vector
        lora=None,                         # subset of ['q','k','v']
        schedule_type: str = "dares_front",
        base_rank: int = 14,
        min_rank: int = 4,
        adalora_max_rank: int = 16,
        adalora_total_rank_budget: int = 144,
    ):
        self.model = model
        self.mode = mode
        self.schedule_type = schedule_type
        self.base_rank = base_rank
        self.min_rank = min_rank
        self.adalora_max_rank = adalora_max_rank
        self.adalora_total_rank_budget = adalora_total_rank_budget
        self.lora = list(lora) if lora is not None else ["q", "v"]

        self.w_As = []
        self.w_Bs = []
        self.lora_modules = []  # list of _LoRA_qkv modules (for potential dynamic updates)
        self.r = r  # may be None, meaning "derive from schedule/mode"

        self.initialize_lora()

    def _derive_rank_vector(self, num_layers: int):
        # If user provided an explicit rank vector, adapt length and return it.
        if self.r is not None:
            r_vec = list(self.r)
            if len(r_vec) < num_layers:
                r_vec = r_vec + [r_vec[-1]] * (num_layers - len(r_vec))
            elif len(r_vec) > num_layers:
                r_vec = r_vec[:num_layers]
            return [max(1, int(v)) for v in r_vec]

        # Mode-specific defaults
        if self.mode == "dares":
            # Original DARES front-heavy pattern (for 12 layers) as default
            default = [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]
            if num_layers <= len(default):
                return default[:num_layers]
            # If there are more layers, just extend with the last value
            return default + [default[-1]] * (num_layers - len(default))

        # For 'schedule' and 'adalora', use a schedule-based vector
        r_vec = _build_rank_vector(num_layers, self.base_rank, self.min_rank, self.schedule_type)

        # For AdaLoRA, enforce a simple global rank budget (static) by scaling
        if self.mode == "adalora" and self.adalora_total_rank_budget > 0:
            total = sum(r_vec)
            if total > self.adalora_total_rank_budget:
                scale = float(self.adalora_total_rank_budget) / float(total)
                new_r = []
                for v in r_vec:
                    vv = max(self.min_rank, int(round(v * scale)))
                    vv = min(vv, self.adalora_max_rank)
                    new_r.append(vv)
                r_vec = new_r
        return r_vec

    def initialize_lora(self):
        # Freeze the backbone; only LoRA and head/other modules are trainable
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        encoder_layers = list(self.model.backbone.encoder.layer)
        num_layers = len(encoder_layers)

        # Compute per-layer target ranks
        rank_vec = self._derive_rank_vector(num_layers)

        for t_layer_i, blk in enumerate(encoder_layers):
            dim = blk.attention.attention.query.in_features
            rank_i = int(rank_vec[t_layer_i])
            rank_i = max(1, rank_i)

            # Decide allocated rank (A/B width) vs. active rank for dynamic (AdaLoRA) case
            if self.mode == "adalora":
                max_rank = max(rank_i, self.adalora_max_rank)
                alloc_rank = max_rank
                active_rank = rank_i  # start with some depth-dependent schedule
            else:
                alloc_rank = rank_i
                active_rank = None  # static Vector-LoRA; use full rank

            # q
            if "q" in self.lora:
                w_q = blk.attention.attention.query
                w_a_linear_q = nn.Linear(dim, alloc_rank, bias=False)
                w_b_linear_q = nn.Linear(alloc_rank, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                lora_q = _LoRA_qkv(w_q, w_a_linear_q, w_b_linear_q, active_rank=active_rank)
                blk.attention.attention.query = lora_q
                self.lora_modules.append(lora_q)

            # v
            if "v" in self.lora:
                w_v = blk.attention.attention.value
                w_a_linear_v = nn.Linear(dim, alloc_rank, bias=False)
                w_b_linear_v = nn.Linear(alloc_rank, dim, bias=False)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                lora_v = _LoRA_qkv(w_v, w_a_linear_v, w_b_linear_v, active_rank=active_rank)
                blk.attention.attention.value = lora_v
                self.lora_modules.append(lora_v)

            # optional 'k'
            if "k" in self.lora:
                w_k = blk.attention.attention.key
                w_a_linear_k = nn.Linear(dim, alloc_rank, bias=False)
                w_b_linear_k = nn.Linear(alloc_rank, dim, bias=False)
                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)
                lora_k = _LoRA_qkv(w_k, w_a_linear_k, w_b_linear_k, active_rank=active_rank)
                blk.attention.attention.key = lora_k
                self.lora_modules.append(lora_k)

        self.reset_parameters()
        print("LoRA params initialized! (mode = {}, schedule = {}, ranks = {})".format(
            self.mode, self.schedule_type, rank_vec
        ))

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    # (Optional) hook for future dynamic updates: you can call this from the Trainer.
    def set_active_ranks(self, per_module_ranks):
        """
        Set active_rank for each _LoRA_qkv module. This enables AdaLoRA-style
        reallocation without changing parameter shapes.
        """
        if len(per_module_ranks) != len(self.lora_modules):
            raise ValueError("Length mismatch: {} modules vs {} ranks".format(
                len(self.lora_modules), len(per_module_ranks)
            ))
        for mod, r in zip(self.lora_modules, per_module_ranks):
            if hasattr(mod, "active_rank"):
                mod.active_rank = int(max(0, r))


# ---------------------
## Commented out code which it replaces
# class LoRAInitializer:
#     def __init__(self, model, r=[14,14,12,12,10,10,8,8,8,8,8,8], lora=['q', 'v']):
#         self.model = model
#         self.r = r
#         self.lora = lora
#         self.w_As = []
#         self.w_Bs = []
#         self.initialize_lora()
#
#     def initialize_lora(self):
#         for param in self.model.backbone.parameters():
#             param.requires_grad = False
#
#         for t_layer_i, blk in enumerate(self.model.backbone.encoder.layer):
#             dim = blk.attention.attention.query.in_features
#
#             if 'q' in self.lora:
#                 w_q = blk.attention.attention.query
#                 w_a_linear_q = nn.Linear(dim, self.r[t_layer_i], bias=False)
#                 w_b_linear_q = nn.Linear(self.r[t_layer_i], dim, bias=False)
#                 self.w_As.append(w_a_linear_q)
#                 self.w_Bs.append(w_b_linear_q)
#                 blk.attention.attention.query = _LoRA_qkv(w_q, w_a_linear_q, w_b_linear_q)
#
#             if 'v' in self.lora:
#                 w_v = blk.attention.attention.value
#                 w_a_linear_v = nn.Linear(dim, self.r[t_layer_i], bias=False)
#                 w_b_linear_v = nn.Linear(self.r[t_layer_i], dim, bias=False)
#                 self.w_As.append(w_a_linear_v)
#                 self.w_Bs.append(w_b_linear_v)
#                 blk.attention.attention.value = _LoRA_qkv(w_v, w_a_linear_v, w_b_linear_v)
#
#             if 'k' in self.lora:
#                 w_k = blk.attention.attention.key
#                 w_a_linear_k = nn.Linear(dim, self.r[t_layer_i], bias=False)
#                 w_b_linear_k = nn.Linear(self.r[t_layer_i], dim, bias=False)
#                 self.w_As.append(w_a_linear_k)
#                 self.w_Bs.append(w_b_linear_k)
#                 blk.attention.attention.key = _LoRA_qkv(w_k, w_a_linear_k, w_b_linear_k)
#
#         self.reset_parameters()
#         print("LoRA params initialized!")
#
#     def reset_parameters(self):
#         for w_A in self.w_As:
#             nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
#         for w_B in self.w_Bs:
#             nn.init.zeros_(w_B.weight)
# -------------------------------


# MODIFICATION: LoRA Schedule/AdaLoRA -----
class DARES(nn.Module):
    def __init__(
        self,
        # LoRA-related knobs (can be wired from MonodepthOptions later)
        lora_mode: str = "dares",              # 'dares', 'schedule', 'adalora'
        lora_schedule_type: str = "dares_front",
        lora_base_rank: int = 14,
        lora_min_rank: int = 4,
        adalora_max_rank: int = 16,
        adalora_total_rank_budget: int = 144,
        r=None,
        lora=None,
    ):
        super(DARES, self).__init__()

        # Load Depth Anything V2 backbone for depth estimation
        model = DepthAnythingForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )

        # Store configuration
        self.config = model.config
        self.backbone = model.backbone
        self.neck = model.neck

        model_head = model.head
        self.head = DepthAnythingDepthEstimationHead(model_head)

        # Resolve LoRA modes / schedule arguments to something concrete
        num_layers = len(self.backbone.encoder.layer)

        if r is not None:
            rank_vec = list(r)
            if len(rank_vec) < num_layers:
                rank_vec = rank_vec + [rank_vec[-1]] * (num_layers - len(rank_vec))
            elif len(rank_vec) > num_layers:
                rank_vec = rank_vec[:num_layers]
        else:
            # Let LoRAInitializer derive it given the mode
            rank_vec = None

        self.lora_mode = lora_mode
        self.lora_schedule_type = lora_schedule_type
        self.lora_base_rank = lora_base_rank
        self.lora_min_rank = lora_min_rank
        self.adalora_max_rank = adalora_max_rank
        self.adalora_total_rank_budget = adalora_total_rank_budget
        self.lora_targets = list(lora) if lora is not None else ["q", "v"]

        # Initialize LoRA parameters (freezes backbone and injects adapters)
        self.lora_initializer = LoRAInitializer(
            model=model,
            mode=self.lora_mode,
            r=rank_vec,
            lora=self.lora_targets,
            schedule_type=self.lora_schedule_type,
            base_rank=self.lora_base_rank,
            min_rank=self.lora_min_rank,
            adalora_max_rank=self.adalora_max_rank,
            adalora_total_rank_budget=self.adalora_total_rank_budget,
        )

        # Expose LoRA A/B weights and decode head for save/load helpers
        self.w_As = self.lora_initializer.w_As
        self.w_Bs = self.lora_initializer.w_Bs
        self.decode_head = self.head  # for compatibility with save_parameters/load_parameters

        # Run any HF post-init hooks
        model.post_init()

    def save_parameters(self, filename: str) -> None:
        r"""Save both LoRA and head parameters to a .pt/.pth file."""
        assert filename.endswith(".pt") or filename.endswith(".pth")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        decode_head_tensors = {}

        # save decode head, only `state_dict`
        if isinstance(self.decode_head, torch.nn.DataParallel) or isinstance(
            self.decode_head, torch.nn.parallel.DistributedDataParallel
        ):
            state_dict = self.decode_head.module.state_dict()
        else:
            state_dict = self.decode_head.state_dict()
        for key, value in state_dict.items():
            decode_head_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **decode_head_tensors}
        torch.save(merged_dict, filename)

        print("saved lora parameters to %s." % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Load both LoRA and head parameters from a .pt/.pth file."""
        assert filename.endswith(".pt") or filename.endswith(".pth")

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            if saved_key not in state_dict:
                continue
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            if saved_key not in state_dict:
                continue
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        decode_head_dict = self.decode_head.state_dict()
        decode_head_keys = list(decode_head_dict.keys())

        # load decode head
        decode_head_values = [state_dict[k] for k in decode_head_keys if k in state_dict]
        decode_head_new_state_dict = {
            k: v for k, v in zip(decode_head_keys, decode_head_values)
        }
        decode_head_dict.update(decode_head_new_state_dict)

        self.decode_head.load_state_dict(decode_head_dict)

        print("loaded lora parameters from %s." % filename)

    def forward(self, pixel_values):
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=None, output_attentions=None
        )
        hidden_states = outputs.feature_maps
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        outputs = {}
        outputs[("disp", 0)] = self.head(hidden_states[3], height, width)
        outputs[("disp", 1)] = self.head(hidden_states[2], height / 2, width / 2)
        outputs[("disp", 2)] = self.head(hidden_states[1], height / 4, width / 4)
        outputs[("disp", 3)] = self.head(hidden_states[0], height / 8, width / 8)
        return outputs


# ---------------------
## Commented out code which it replaces
# class DARES(nn.Module):
#     def __init__(self, r = [14,14,12,12,10,10,8,8,8,8,8,8], lora = ['q', 'v']):
#         super(DARES, self).__init__()
#         model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
#         self.r = r
#         self.lora = lora
#         self.config = model.config
#         self.backbone = model.backbone
#
#         # Initialize LoRA parameters
#         self.lora_initializer = LoRAInitializer(model, r, lora)
#
#         self.neck = model.neck
#         model_head = model.head
#         self.head = DepthAnythingDepthEstimationHead(model_head)
#         model.post_init()
#
#     def save_parameters(self, filename: str) -> None:
#         r"""Only safetensors is supported now.
#
#         pip install safetensor if you do not have one installed yet.
#
#         save both lora and fc parameters.
#         """
#
#         assert filename.endswith(".pt") or filename.endswith('.pth')
#
#         num_layer = len(self.w_As)  # actually, it is half
#         a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
#         b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
#         decode_head_tensors = {}
#
#         # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
#         if isinstance(self.decode_head, torch.nn.DataParallel) or isinstance(self.decode_head, torch.nn.parallel.DistributedDataParallel):
#             state_dict = self.decode_head.module.state_dict()
#         else:
#             state_dict = self.decode_head.state_dict()
#         for key, value in state_dict.items():
#             decode_head_tensors[key] = value
#
#         merged_dict = {**a_tensors, **b_tensors, **decode_head_tensors}
#         torch.save(merged_dict, filename)
#
#         print('saved lora parameters to %s.' % filename)
#
#     def load_parameters(self, filename: str, device: str) -> None:
#         r"""Only safetensors is supported now.
#
#         pip install safetensor if you do not have one installed yet.\
#
#         load both lora and fc parameters.
#         """
#
#         assert filename.endswith(".pt") or filename.endswith('.pth')
#
#         state_dict = torch.load(filename, map_location=device)
#
#         for i, w_A_linear in enumerate(self.w_As):
#             saved_key = f"w_a_{i:03d}"
#             saved_tensor = state_dict[saved_key]
#             w_A_linear.weight = Parameter(saved_tensor)
#
#         for i, w_B_linear in enumerate(self.w_Bs):
#             saved_key = f"w_b_{i:03d}"
#             saved_tensor = state_dict[saved_key]
#             w_B_linear.weight = Parameter(saved_tensor)
#
#         decode_head_dict = self.decode_head.state_dict()
#         decode_head_keys = decode_head_dict.keys()
#
#         # load decode head
#         decode_head_keys = [k for k in decode_head_keys]
#         decode_head_values = [state_dict[k] for k in decode_head_keys]
#         decode_head_new_state_dict = {k: v for k, v in zip(decode_head_keys, decode_head_values)}
#         decode_head_dict.update(decode_head_new_state_dict)
#
#         self.decode_head.load_state_dict(decode_head_dict)
#
#         print('loaded lora parameters from %s.' % filename)
#
#     def forward(self, pixel_values):
#         outputs = self.backbone.forward_with_filtered_kwargs(
#             pixel_values, output_hidden_states=None, output_attentions=None
#         )
#         hidden_states = outputs.feature_maps
#         _, _, height, width = pixel_values.shape
#         patch_size = self.config.patch_size
#         patch_height = height // patch_size
#         patch_width = width // patch_size
#         hidden_states = self.neck(hidden_states, patch_height, patch_width)
#         outputs = {}
#         outputs[("disp", 0)] = self.head(hidden_states[3], height, width)
#         outputs[("disp", 1)] = self.head(hidden_states[2], height/2, width/2)
#         outputs[("disp", 2)] = self.head(hidden_states[1], height/4, width/4)
#         outputs[("disp", 3)] = self.head(hidden_states[0], height/8, width/8)
#         return outputs
# -------------------------------
