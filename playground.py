from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ResizeKeepRatioPad:
    """Replica of training keepratio resize: aspect-preserving resize + padding."""

    def __init__(self, size_tuple=(256, 256), fill=-1):
        self.target_h, self.target_w = size_tuple
        self.fill = float(fill)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim != 3:
            raise ValueError(f"Expected (C,H,W), got shape={tuple(img.shape)}")
        if not img.is_floating_point():
            img = img.float()

        c, h, w = img.shape
        scale = min(self.target_h / max(h, 1), self.target_w / max(w, 1))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
        resized = resized.squeeze(0)

        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=self.fill)


INPUT_PATH = Path("tmp/1PA177.nii_z0007.npy")
KEEP_RATIO_SIZE = (512, 512)  # Matches training keepratio resize target (H, W)
ORIGINAL_BG_OUT = INPUT_PATH.with_name(f"{INPUT_PATH.stem}_background.png")
KEEP_RATIO_BG_OUT = INPUT_PATH.with_name(
    f"{INPUT_PATH.stem}_background_keepratio{KEEP_RATIO_SIZE[0]}.png"
)
BACKGROUND_SENTINEL = -1.0
EPSILON = 1e-3  # accommodate minor interpolation drift around the sentinel


def extract_background_mask(array: np.ndarray) -> np.ndarray:
    mask = np.isclose(array, BACKGROUND_SENTINEL, atol=EPSILON)
    if not mask.any():
        mask = array <= (BACKGROUND_SENTINEL + EPSILON)
    return mask


def mask_to_image(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 0, 255).astype(np.uint8)


def save_mask_image(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def apply_keep_ratio_resize(array: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    resized = ResizeKeepRatioPad(size_tuple=KEEP_RATIO_SIZE, fill=BACKGROUND_SENTINEL)(tensor)
    return resized.squeeze(0).detach().cpu().numpy()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    original = np.load(INPUT_PATH).astype(np.float32)

    original_mask = extract_background_mask(original)
    save_mask_image(mask_to_image(original_mask), ORIGINAL_BG_OUT)

    resized = apply_keep_ratio_resize(original)
    resized_mask = extract_background_mask(resized)
    save_mask_image(mask_to_image(resized_mask), KEEP_RATIO_BG_OUT)

    print(f"Saved original-size background mask to: {ORIGINAL_BG_OUT}")
    print(f"Saved keepratio background mask to: {KEEP_RATIO_BG_OUT}")


if __name__ == "__main__":
    main()
