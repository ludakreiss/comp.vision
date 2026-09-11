"""
Grad-CAM for the fusion DeepfakeModel (EfficientNet-B0 RGB branch +
Multi-Scale SRM frequency branch + Spatial-Frequency Cross-Attention).

Target layer: `model.sfca` (SpatialFrequencyCrossAttention).

Why this layer and not `model.rgb_backbone[-1]`: the classifier head consumes
`avg_pool(attended_rgb_map)` concatenated with `avg_pool(freq_map)`, where
`attended_rgb_map` is exactly the output of `model.sfca`. Hooking the raw
EfficientNet backbone output would show what the RGB branch saw *before* the
frequency cross-attention modulates it, missing the fusion step the model
actually bases its RGB-side pooled features on. `model.sfca`'s output is
verified (see model.py forward()) to be the last spatial tensor feeding the
classifier via global average pooling, with an intact gradient path back to
the input pixels.

This is a from-scratch implementation against the current model.py -- it does
not reuse or import any prior/removed Grad-CAM code.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config


class GradCAM:
    """Grad-CAM via forward/backward hooks on a single target layer.

    Usage:
        with GradCAM(model) as cam_gen:
            cam, fake_logit = cam_gen.generate(input_tensor)
    """

    def __init__(self, model, target_layer=None):
        if target_layer is None:
            target_layer = getattr(model, "sfca", None)
            if target_layer is None:
                raise ValueError(
                    "Model has no `sfca` module (branch_mode/model_variant is not "
                    "'fusion'); pass target_layer explicitly for this architecture."
                )
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._fwd_handle = None
        self._bwd_handle = None

    def _save_activation(self, module, inputs, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def __enter__(self):
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
        self._fwd_handle = None
        self._bwd_handle = None
        self._activations = None
        self._gradients = None
        return False

    def generate(self, input_tensor):
        """
        input_tensor: [1, 3, H, W], preprocessed exactly like inference (resized +
        ImageNet-normalized).

        Returns:
            cam: np.ndarray [H, W] float32 in [0, 1], upsampled to input H/W.
            fake_logit: python float, the raw fake-class logit used as the
                Grad-CAM target (this model has a single binary logit -- there
                is no second class to contrast against).
        """
        was_training = self.model.training
        self.model.eval()

        x = input_tensor.clone().detach()
        x.requires_grad_(True)

        self._activations = None
        self._gradients = None

        # Grad-CAM needs a live autograd graph. Must not be called from inside
        # an outer torch.inference_mode() block -- inference-mode tensors can
        # never re-enter autograd, even under enable_grad().
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            output = self.model(x)
            logits = output["deepfake_logit"] if isinstance(output, dict) else output[0]
            score = logits.squeeze()
            score.backward()

        if was_training:
            self.model.train()

        activations = self._activations
        gradients = self._gradients
        if activations is None or gradients is None:
            raise RuntimeError(
                "Grad-CAM did not capture activations/gradients -- the target "
                "layer may not be on the path from input to the fake logit."
            )

        # alpha_k = spatial average of d(score)/d(A_k) ; CAM = ReLU(sum_k alpha_k * A_k)
        alpha = gradients.mean(dim=(2, 3), keepdim=True)                  # [B, C, 1, 1]
        cam = F.relu((alpha * activations).sum(dim=1, keepdim=True))      # [B, 1, h, w]
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0]  # [H, W]

        cam_min = cam.min()
        cam_max = cam.max()
        denom = (cam_max - cam_min).clamp_min(1e-8)
        cam = (cam - cam_min) / denom

        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        cam_np = np.nan_to_num(cam_np, nan=0.0, posinf=1.0, neginf=0.0)
        fake_logit = float(score.detach().item())

        # Clear parameter gradients from this pass so they don't linger between calls.
        self.model.zero_grad(set_to_none=True)

        return cam_np, fake_logit


def overlay_heatmap(base_image, cam, alpha=0.45, colormap="jet"):
    """
    base_image: PIL.Image (RGB).
    cam: np.ndarray [H, W] in [0, 1].
    Returns a PIL.Image (RGB) with the heatmap alpha-blended over base_image.
    """
    import matplotlib

    base = base_image.convert("RGB")
    w, h = base.size

    if cam.shape != (h, w):
        cam_t = torch.from_numpy(cam)[None, None]
        cam_t = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam_t[0, 0].numpy()

    cam = np.clip(cam, 0.0, 1.0)
    try:
        cmap = matplotlib.colormaps[colormap]
    except AttributeError:
        cmap = matplotlib.cm.get_cmap(colormap)
    colored = cmap(cam)[:, :, :3]  # RGBA -> RGB, floats in [0, 1]
    heatmap_rgb = (colored * 255.0).astype(np.float32)

    base_np = np.array(base).astype(np.float32)
    blended = (1.0 - alpha) * base_np + alpha * heatmap_rgb
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended, mode="RGB")


def compute_gradcam_overlay(model, eval_transform, pil_image, alpha=0.45):
    """
    High-level entry point for a Gradio callback: preprocess `pil_image` the
    same way as normal inference, run Grad-CAM against the model's fake logit,
    and return an overlay image ready to display.

    Returns (overlay_image: PIL.Image RGB, fake_logit: float).
    """
    rgb_image = pil_image.convert("RGB")
    tensor = eval_transform(rgb_image).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with GradCAM(model) as cam_gen:
        cam, fake_logit = cam_gen.generate(tensor)

    # Resize the base image to match the model's actual input resolution so the
    # overlay lines up pixel-for-pixel with the CAM's spatial reference frame.
    resized_base = rgb_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC)
    overlay = overlay_heatmap(resized_base, cam, alpha=alpha)

    return overlay, fake_logit
