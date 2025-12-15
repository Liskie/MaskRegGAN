#!/usr/bin/env python3

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Suppress noisy third-party deprecation warnings we cannot control here.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated since 0.13",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
    category=UserWarning,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from scipy import linalg
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from trainer.datasets import ValDataset
from trainer.reg import Reg
from trainer.transformer import Transformer_2D
from trainer.utils import (
    Resize,
    ResizeKeepRatioPad,
    ToTensor,
    compute_mae,
    compute_psnr,
    compute_ssim,
    compose_slice_name,
    to_uint8_image,
)
from trainer.CycTrainerFusion import SharedBackboneGenerator

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except ImportError:
    LearnedPerceptualImagePatchSimilarity = None

RICH_CONSOLE = Console(force_terminal=True)


def load_config(path: str) -> Dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def resolve_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_rd_mode(config: Dict) -> Optional[str]:
    mode = config.get("rd_mode", config.get("rd_input_type", "none"))
    if mode is None:
        return None
    mode = str(mode).strip().lower()
    if mode in ("none", "off", "false", "0", ""):
        return None
    if mode in ("mask", "weights"):
        return mode
    raise ValueError(f"Unsupported rd_mode '{mode}'")


def resolve_dataset_root(config: Dict, override: Optional[str]) -> str:
    if override is not None:
        return str(Path(str(override)).expanduser().resolve())
    for key in ("test_dataroot", "val_dataroot", "dataroot"):
        cand = config.get(key)
        if cand:
            return str(Path(str(cand)).expanduser().resolve())
    trainer_name = str(config.get("trainer", "")).lower()
    if "nirvis" in trainer_name or str(config.get("nirvis_root", "")).strip() != "":
        base = config.get("nirvis_root") or config.get("dataroot")
        split_candidates = [
            config.get("nirvis_val_split"),
            config.get("nirvis_train_split"),
            "test",
            "val",
            "train",
        ]
        for split in split_candidates:
            if base and split:
                return str(Path(str(base)).expanduser().joinpath(str(split)).resolve())
    raise KeyError(
        "Dataset root not specified; provide --data-root or set test_dataroot/val_dataroot/dataroot in config."
    )


def _resolve_dir_with_base(base: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path) if path.exists() else None
    if path.exists():
        return str(path.resolve())
    base_path = Path(base) if base else None
    if base_path:
        candidate = base_path / path
        if candidate.exists():
            return str(candidate.resolve())
        parent_candidate = base_path.parent / path
        if parent_candidate.exists():
            return str(parent_candidate.resolve())
    return None


def _guess_nirvis_dirs(config: Dict, dataset_root: str) -> Tuple[Optional[str], Optional[str]]:
    root = Path(dataset_root)
    if not root.exists():
        return None, None
    nir_dir_name = config.get("nirvis_nir_dir", "NIR")
    vis_dir_name = config.get("nirvis_vis_dir", "RGB")
    direction = str(config.get("nirvis_direction", "nir2vis")).lower()

    def _ordered(nir_dir: Path, vis_dir: Path) -> Tuple[Optional[str], Optional[str]]:
        if not nir_dir.is_dir() or not vis_dir.is_dir():
            return None, None
        if direction == "vis2nir":
            return str(vis_dir), str(nir_dir)
        return str(nir_dir), str(vis_dir)

    nir_dir, vis_dir = _ordered(root / nir_dir_name, root / vis_dir_name)
    if nir_dir and vis_dir:
        return nir_dir, vis_dir

    split_candidates = [
        config.get("nirvis_val_split"),
        config.get("nirvis_train_split"),
        "test",
        "val",
        "validation",
        "train",
    ]
    seen_splits = set()
    for split in split_candidates:
        if not split or split in seen_splits:
            continue
        seen_splits.add(split)
        nir_dir, vis_dir = _ordered(root / split / nir_dir_name, root / split / vis_dir_name)
        if nir_dir and vis_dir:
            return nir_dir, vis_dir

    nir_guess = vis_guess = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if nir_guess is None and "nir" in name:
            nir_guess = child
        if vis_guess is None and ("vis" in name or "rgb" in name):
            vis_guess = child
    if nir_guess and vis_guess:
        nir_dir, vis_dir = _ordered(nir_guess, vis_guess)
        if nir_dir and vis_dir:
            return nir_dir, vis_dir
    return None, None


def _find_existing_dir(base: Path, rel: Optional[str]) -> Optional[str]:
    if not rel:
        return None
    rel_path = Path(rel)
    candidates = []
    if rel_path.is_absolute():
        candidates.append(rel_path)
    else:
        candidates.append(base / rel_path)
        candidates.append(base / rel_path.name)
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _guess_saropt_dirs(config: Dict, dataset_root: str) -> Tuple[Optional[str], Optional[str]]:
    root = Path(dataset_root)
    opt_dir = None
    sar_dir = None
    opt_candidates = [
        config.get("test_domain_opt_dir"),
        config.get("saropt_test_opt_dir"),
        config.get("val_domain_opt_dir"),
        config.get("saropt_val_opt_dir"),
        config.get("saropt_opt_dir"),
    ]
    sar_candidates = [
        config.get("test_domain_sar_dir"),
        config.get("saropt_test_sar_dir"),
        config.get("val_domain_sar_dir"),
        config.get("saropt_val_sar_dir"),
        config.get("saropt_sar_dir"),
    ]
    for cand in opt_candidates:
        opt_dir = _find_existing_dir(root, cand)
        if opt_dir:
            break
    for cand in sar_candidates:
        sar_dir = _find_existing_dir(root, cand)
        if sar_dir:
            break
    children = [child for child in root.iterdir() if child.is_dir()]
    if opt_dir is None:
        for child in children:
            if "opt" in child.name.lower():
                opt_dir = str(child)
                break
    if sar_dir is None:
        for child in children:
            if "sar" in child.name.lower():
                sar_dir = str(child)
                break
    return opt_dir, sar_dir


def _initialize_inception_extractor(device: torch.device):
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    ).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return inception, preprocess


def _compute_inception_activations(batch: torch.Tensor,
                                   model: torch.nn.Module,
                                   preprocess,  # torchvision transform
                                   device: torch.device) -> np.ndarray:
    imgs = torch.clamp((batch.detach().cpu() + 1.0) * 0.5, 0.0, 1.0)
    processed = []
    for img in imgs:
        processed.append(preprocess(img))
    proc = torch.stack(processed, dim=0).to(device)
    with torch.no_grad():
        feats = model(proc)
    return feats.detach().cpu().numpy()


def _calculate_fid_from_activations(fake_acts: np.ndarray, real_acts: np.ndarray) -> float:
    mu1, sigma1 = fake_acts.mean(axis=0), np.cov(fake_acts, rowvar=False)
    mu2, sigma2 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def build_eval_loader(config: Dict, dataset_root: str) -> DataLoader:
    mode = str(config.get("resize_mode", "resize")).lower()
    if mode not in ("resize", "keepratio"):
        mode = "resize"
    if mode == "keepratio":
        last_tf = ResizeKeepRatioPad(size_tuple=(config["size"], config["size"]), fill=-1)
    else:
        last_tf = Resize(size_tuple=(config["size"], config["size"]))
    transforms = [ToTensor(), last_tf]
    rd_mode = resolve_rd_mode(config)
    trainer_name = str(config.get("trainer", "")).lower()
    domain_a_dir = _resolve_dir_with_base(
        dataset_root,
        config.get("test_domain_a_dir")
        or config.get("val_domain_a_dir")
        or config.get("domain_a_dir"),
    )
    domain_b_dir = _resolve_dir_with_base(
        dataset_root,
        config.get("test_domain_b_dir")
        or config.get("val_domain_b_dir")
        or config.get("domain_b_dir"),
    )
    domain_a_channels = (
        config.get("test_domain_a_channels")
        or config.get("val_domain_a_channels")
        or config.get("domain_a_channels")
    )
    domain_b_channels = (
        config.get("test_domain_b_channels")
        or config.get("val_domain_b_channels")
        or config.get("domain_b_channels")
    )
    if (
        (domain_a_dir is None or domain_b_dir is None)
        and (
            "saropt" in trainer_name
            or str(config.get("saropt_direction", "")).strip() != ""
            or str(config.get("saropt_root", "")).strip() != ""
        )
    ):
        direction = str(config.get("saropt_direction", "")).lower()
        opt_dir_guess, sar_dir_guess = _guess_saropt_dirs(config, dataset_root)
        if direction == "sar2rgb":
            domain_a_dir = domain_a_dir or sar_dir_guess
            domain_b_dir = domain_b_dir or opt_dir_guess
        elif direction == "rgb2sar":
            domain_a_dir = domain_a_dir or opt_dir_guess
            domain_b_dir = domain_b_dir or sar_dir_guess
        else:
            domain_a_dir = domain_a_dir or opt_dir_guess
            domain_b_dir = domain_b_dir or sar_dir_guess
        if domain_a_dir is None or domain_b_dir is None:
            raise ValueError(
                "Unable to resolve SAROPT domain directories. "
                "Set test_domain_a_dir/test_domain_b_dir or ensure dataset_root contains 'opt' and 'sar' folders."
            )
    if (domain_a_dir is None or domain_b_dir is None) and (
        "nirvis" in trainer_name
        or str(config.get("nirvis_root", "")).strip() != ""
        or str(config.get("nirvis_direction", "")).strip() != ""
    ):
        nir_a_dir, nir_b_dir = _guess_nirvis_dirs(config, dataset_root)
        domain_a_dir = domain_a_dir or nir_a_dir
        domain_b_dir = domain_b_dir or nir_b_dir
    if "nirvis" in trainer_name or str(config.get("nirvis_root", "")).strip() != "":
        direction = str(config.get("nirvis_direction", "nir2vis")).lower()
        if direction == "vis2nir":
            domain_a_channels = domain_a_channels or 3
            domain_b_channels = domain_b_channels or 1
        else:
            domain_a_channels = domain_a_channels or 1
            domain_b_channels = domain_b_channels or 3
    dataset = ValDataset(
        root=dataset_root,
        transforms_=transforms,
        unaligned=False,
        rd_input_type=rd_mode,
        rd_mask_dir=config.get("rd_mask_dir", ""),
        rd_weights_dir=config.get("rd_weights_dir", ""),
        rd_w_min=float(config.get("rd_w_min", 0.0)),
        cache_mode=config.get("cache_mode", "mmap"),
        rd_cache_weights=bool(config.get("rd_cache_weights", False)),
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        domain_a_channels=domain_a_channels,
        domain_b_channels=domain_b_channels,
    )
    num_workers = int(config.get("n_cpu", 0))
    kwargs = dict(
        batch_size=int(config.get("batchSize", 1)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(config.get("cuda", False)),
        timeout=int(config.get("dataloader_timeout", 0)),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = max(1, int(config.get("prefetch_factor", 2)))
    return DataLoader(dataset, **kwargs)


def squeeze_hw(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    # Drop leading batch dimension(s) but preserve channel info.
    while out.ndim > 3 and out.shape[0] == 1:
        out = np.squeeze(out, axis=0)
    out = np.squeeze(out)
    if out.ndim == 2:
        return out.astype(np.float32, copy=False)
    if out.ndim == 3:
        if out.shape[0] in (1, 3):
            return out.astype(np.float32, copy=False)
        if out.shape[-1] in (1, 3):
            return out.transpose(2, 0, 1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported slice shape {out.shape}; expected HxW or CxHxW.")


def build_keep_mask(
    real: np.ndarray,
    rd_weight_batch: Optional[np.ndarray],
    index: int,
) -> np.ndarray:
    body = (real != -1).astype(np.float32)
    if body.ndim == 3:
        if body.shape[0] in (1, 3):
            body = np.min(body, axis=0)
        elif body.shape[-1] in (1, 3):
            body = np.min(body, axis=-1)
        else:
            body = body.mean(axis=0)
    if rd_weight_batch is None:
        return body
    weight = np.squeeze(rd_weight_batch[index]).astype(np.float32)
    if weight.ndim == 3:
        if weight.shape[0] == 1:
            weight = weight[0]
        elif weight.shape[-1] == 1:
            weight = weight[..., 0]
        else:
            weight = weight.mean(axis=0)
    if weight.shape != body.shape:
        weight = cv2.resize(
            weight.astype(np.float32),
            (body.shape[1], body.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    weight = np.clip(weight, 0.0, 1.0)
    keep = body * weight
    if np.sum(keep) <= 0:
        return body
    return keep


def prepare_images_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
    img = tensor.detach()
    if img.ndim != 4:
        raise ValueError(f"Expected BCHW tensor for metrics, got {img.shape}")
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)
    img = torch.clamp((img + 1.0) * 0.5, 0.0, 1.0)
    return img


def _numpy_to_bchw(arr: np.ndarray) -> torch.Tensor:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim == 2:
        data = data[None, None, ...]
    elif data.ndim == 3:
        if data.shape[0] in (1, 3):
            data = data[None, ...]
        elif data.shape[-1] in (1, 3):
            data = data.transpose(2, 0, 1)[None, ...]
        else:
            raise ValueError(f"Unsupported slice shape {data.shape} for conversion to BCHW.")
    else:
        raise ValueError(f"Unsupported array shape {data.shape} for conversion to BCHW.")
    return torch.from_numpy(data)


def load_fusion_generator(
    config: Dict,
    checkpoint: Path,
    device: torch.device,
) -> SharedBackboneGenerator:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "netG_A2B" in state:
        state = state["netG_A2B"]
    if isinstance(state, dict):
        sample_key = next(iter(state.keys()))
        if sample_key.startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}
    n_heads = int(config.get("fusion_heads", 3))
    upsample_mode = str(config.get("generator_upsample_mode", "resize")).lower()
    net = SharedBackboneGenerator(
        input_nc=config["input_nc"],
        output_nc=config["output_nc"],
        n_residual_blocks=config.get("fusion_residual_blocks", 9),
        n_heads=n_heads,
        upsample_mode=upsample_mode,
        share_body=bool(config.get("fusion_share_body", True)),
    ).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def _resolve_reg_channels(config: Dict) -> Tuple[int, int]:
    fake_nc = int(config.get("reg_fake_nc", config.get("output_nc", config.get("input_nc", 1))))
    real_nc = int(config.get("reg_real_nc", config.get("output_nc", config.get("input_nc", 1))))
    return fake_nc, real_nc


def load_registration(
    config: Dict,
    checkpoint: Optional[Path],
    device: torch.device,
) -> Tuple[Optional[Reg], Optional[Transformer_2D]]:
    if not (config.get("regist") and config.get("eval_with_registration")):
        return None, None
    if checkpoint is None or not checkpoint.is_file():
        print(f"[warn] Registration weights not found at {checkpoint}; skipping registration.")
        return None, None
    fake_nc, real_nc = _resolve_reg_channels(config)
    reg_net = Reg(config["size"], config["size"], fake_nc, real_nc).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "R_A" in state:
        state = state["R_A"]
    if isinstance(state, dict):
        sample_key = next(iter(state.keys()))
        if sample_key.startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}
    reg_net.load_state_dict(state)
    reg_net.eval()
    transformer = Transformer_2D().to(device)
    return reg_net, transformer


class EnsembleAccumulator:
    def __init__(self, expected_models: int):
        self.expected = expected_models
        self.store: Dict[str, Dict] = {}

    def add(
        self,
        slice_name: str,
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray],
        meta: Dict,
    ) -> None:
        entry = self.store.setdefault(
            slice_name,
            {
                "sum": np.zeros_like(pred, dtype=np.float64),
                "count": 0,
                "target": target.astype(np.float32, copy=False),
                "mask": mask.astype(np.float32, copy=False) if mask is not None else None,
                "meta": meta,
            },
        )
        entry["sum"] += pred.astype(np.float64, copy=False)
        entry["count"] += 1

    def finalize(self) -> Dict[str, Dict]:
        for name, entry in self.store.items():
            if entry["count"] != self.expected:
                raise RuntimeError(f"Slice '{name}' has {entry['count']} predictions (expected {self.expected}).")
        return self.store


def evaluate_fusion_model(
    config: Dict,
    loader: DataLoader,
    device: torch.device,
    generator_path: Path,
    reg_path: Optional[Path],
    compute_fid: bool,
    compute_lpips: bool,
    metrics_use_rd: bool,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Dict], EnsembleAccumulator, Optional[float]]:
    generator = load_fusion_generator(config, generator_path, device)
    reg_net, transformer = load_registration(config, reg_path, device)
    n_heads = generator.n_heads
    head_ids = list(range(n_heads))
    results = {
        head: {"mae_sum": 0.0, "psnr_sum": 0.0, "ssim_sum": 0.0, "count": 0}
        for head in head_ids
    }
    fid_model = None
    fid_preprocess = None
    fid_real_store: List[np.ndarray] = []
    fid_fake_store: Dict[int, List[np.ndarray]] = {}
    fid_ensemble_store: List[np.ndarray] = []
    lpips_modules = {}
    if compute_fid:
        fid_model, fid_preprocess = _initialize_inception_extractor(device)
        fid_fake_store = {head: [] for head in head_ids}
    if compute_lpips:
        if LearnedPerceptualImagePatchSimilarity is None:
            raise RuntimeError("LPIPS metric not available; install torchmetrics[image].")
        lpips_net = str(config.get("_lpips_net", "vgg"))
        for head in head_ids:
            lpips_modules[head] = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net, normalize=True).to(device)

    viz_entries: Dict[str, Dict[str, np.ndarray]] = {}
    ensemble = EnsembleAccumulator(expected_models=len(head_ids))

    torch.set_grad_enabled(False)
    try:
        total_slices = len(loader.dataset)  # type: ignore[attr-defined]
    except Exception:
        total_slices = None
    progress = Progress(
        TextColumn("[bold blue]fusion[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
        transient=False,
    )
    with progress:
        task_id = progress.add_task(
            "eval",
            total=total_slices if (isinstance(total_slices, int) and total_slices > 0) else None,
        )
        for batch_idx, batch in enumerate(loader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            outputs = generator(real_A, head_indices=head_ids)
            eval_outputs = {}
            if reg_net is not None and transformer is not None:
                for head in head_ids:
                    flow = reg_net(outputs[head], real_B)
                    eval_outputs[head] = transformer(outputs[head], flow)
            else:
                eval_outputs = outputs

            real_np = real_B.detach().cpu().numpy()
            rd_weight = batch.get("rd_weight")
            if isinstance(rd_weight, torch.Tensor):
                rd_weight_np = rd_weight.detach().cpu().numpy()
            else:
                rd_weight_np = None

            # Metrics modules update
            if fid_model is not None or lpips_modules:
                real_for_metrics = prepare_images_for_metrics(real_B)
                if fid_model is not None and fid_preprocess is not None:
                    real_feats = _compute_inception_activations(real_B, fid_model, fid_preprocess, device)
                    fid_real_store.append(real_feats)
                for head in head_ids:
                    fake_imgs = prepare_images_for_metrics(eval_outputs[head])
                    if fid_model is not None and fid_preprocess is not None:
                        fake_feats = _compute_inception_activations(eval_outputs[head], fid_model, fid_preprocess, device)
                        fid_fake_store[head].append(fake_feats)
                    if head in lpips_modules:
                        lpips_val = lpips_modules[head](fake_imgs, real_for_metrics)
                        results[head].setdefault("lpips_sum", 0.0)
                        results[head].setdefault("lpips_count", 0.0)
                        results[head]["lpips_sum"] += float(lpips_val.sum().item())
                        results[head]["lpips_count"] += float(lpips_val.numel())

            batch_count = real_np.shape[0]
            for b in range(batch_count):
                slice_name = compose_slice_name(batch, batch_idx, b)
                real_2d = squeeze_hw(real_np[b])
                keep_mask = None
                if metrics_use_rd:
                    keep_mask = build_keep_mask(real_2d, rd_weight_np, b)

                if slice_name not in viz_entries:
                    viz_entries[slice_name] = {}
                    viz_entries[slice_name]["input"] = squeeze_hw(batch["A"].detach().cpu().numpy()[b])
                    viz_entries[slice_name]["target"] = real_2d

                meta = {
                    "patient_id": _safe_index(batch.get("patient_id"), b),
                    "slice_id": _safe_index(batch.get("slice_id"), b),
                }

                for head in head_ids:
                    fake_2d = squeeze_hw(eval_outputs[head].detach().cpu().numpy()[b])
                    mae = compute_mae(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                    psnr = compute_psnr(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                    ssim = compute_ssim(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                    results[head]["mae_sum"] += mae
                    results[head]["psnr_sum"] += psnr
                    results[head]["ssim_sum"] += ssim
                    results[head]["count"] += 1
                    viz_entries[slice_name][f"head{head}"] = fake_2d
                    ensemble.add(
                        slice_name,
                        pred=fake_2d,
                        target=real_2d,
                        mask=keep_mask if metrics_use_rd else None,
                        meta=meta,
                    )
            if fid_model is not None and fid_preprocess is not None:
                avg_pred = torch.stack([eval_outputs[h] for h in head_ids], dim=0).mean(dim=0)
                fid_ensemble_store.append(_compute_inception_activations(avg_pred, fid_model, fid_preprocess, device))
            progress.advance(task_id, advance=batch_count)

    ensemble_fid_value = None
    if fid_model is not None:
        if fid_real_store:
            real_acts = np.concatenate(fid_real_store, axis=0)
            for head in head_ids:
                fake_list = fid_fake_store.get(head, [])
                if fake_list:
                    fake_acts = np.concatenate(fake_list, axis=0)
                    results[head]["fid"] = _calculate_fid_from_activations(fake_acts, real_acts)
            if fid_ensemble_store:
                ensemble_fake = np.concatenate(fid_ensemble_store, axis=0)
                ensemble_fid_value = _calculate_fid_from_activations(ensemble_fake, real_acts)
        else:
            print("[warn] FID requested but no activations were collected.")

    for head in head_ids:
        if results[head]["count"] == 0:
            raise RuntimeError(f"Head {head} produced no samples.")
        if head in lpips_modules and results[head].get("lpips_count", 0.0) > 0:
            results[head]["lpips"] = results[head]["lpips_sum"] / results[head]["lpips_count"]
        results[head]["mae"] = results[head]["mae_sum"] / results[head]["count"]
        results[head]["psnr"] = results[head]["psnr_sum"] / results[head]["count"]
        results[head]["ssim"] = results[head]["ssim_sum"] / results[head]["count"]

    return results, viz_entries, ensemble, ensemble_fid_value


def _safe_index(seq: Optional[Iterable], idx: int):
    if seq is None:
        return None
    if isinstance(seq, (list, tuple)):
        return seq[idx] if idx < len(seq) else None
    return seq


def compute_ensemble_metrics(
    store: EnsembleAccumulator,
    metrics_use_rd: bool,
    device: torch.device,
    compute_fid: bool = False,
    compute_lpips: bool = False,
    fid_feature: int = 64,
    lpips_net: str = "vgg",
    precomputed_fid: Optional[float] = None,
) -> Dict[str, float]:
    entries = store.finalize()
    mae_sum = psnr_sum = ssim_sum = 0.0
    count = 0
    lpips_metric = None
    lpips_sum = lpips_count = 0.0
    if compute_fid and precomputed_fid is None:
        raise RuntimeError("Internal error: ensemble FID should be precomputed.")
    if compute_lpips:
        if LearnedPerceptualImagePatchSimilarity is None:
            raise RuntimeError("LPIPS metric not available; install torchmetrics[image].")
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net, normalize=True).to(device)
    for entry in entries.values():
        avg_pred = (entry["sum"] / entry["count"]).astype(np.float32)
        mask = entry["mask"] if metrics_use_rd else None
        target = entry["target"]
        mae_sum += compute_mae(avg_pred, target, mask=mask)
        psnr_sum += compute_psnr(avg_pred, target, mask=mask)
        ssim_sum += compute_ssim(avg_pred, target, mask=mask)
        count += 1
        if lpips_metric is not None:
            fake_tensor = _numpy_to_bchw(avg_pred).to(device)
            real_tensor = _numpy_to_bchw(target).to(device)
            fake_imgs = prepare_images_for_metrics(fake_tensor)
            real_imgs = prepare_images_for_metrics(real_tensor)
            if lpips_metric is not None:
                lpips_val = lpips_metric(fake_imgs, real_imgs)
                lpips_sum += float(lpips_val.sum().item())
                lpips_count += float(lpips_val.numel())
    if count == 0:
        raise RuntimeError("No slices available for ensemble metrics.")
    metrics = {
        "mae": mae_sum / count,
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
        "slices": count,
    }
    if precomputed_fid is not None:
        metrics["fid"] = float(precomputed_fid)
    if lpips_metric is not None and lpips_count > 0:
        metrics["lpips"] = lpips_sum / lpips_count
    return metrics


def save_slice_figure(
    slice_name: str,
    entry: Dict[str, np.ndarray],
    head_ids: List[int],
    out_dir: Path,
) -> None:
    input_arr = entry["input"]
    target_arr = entry["target"]
    pred_arrays = [entry[f"head{h}"] for h in head_ids]
    avg_arr = np.mean(np.stack(pred_arrays, axis=0), axis=0).astype(np.float32)
    images = (
        [to_uint8_image(input_arr), to_uint8_image(target_arr)]
        + [to_uint8_image(p) for p in pred_arrays]
        + [to_uint8_image(avg_arr)]
    )
    titles = ["Input", "Target"] + [f"Head {h}" for h in head_ids] + ["Average"]
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3), constrained_layout=True)
    for ax, img, title in zip(axes, images, titles):
        if img is None:
            ax.axis("off")
            continue
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    out_path = out_dir / f"{slice_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_visualizations(
    entries: Dict[str, Dict[str, np.ndarray]],
    head_ids: List[int],
    out_dir: Path,
    max_figs: int = 0,
    workers: int = 0,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    names = sorted(entries.keys())
    if isinstance(max_figs, int) and max_figs > 0:
        names = names[:max_figs]
    total = len(names)
    if total == 0:
        return 0
    progress = Progress(
        TextColumn("[magenta]figures[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
        transient=False,
    )
    with progress:
        task_id = progress.add_task("save", total=total)
        if isinstance(workers, int) and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for slice_name in names:
                    entry = entries[slice_name]
                    if "input" in entry and "target" in entry and not any(f"head{h}" not in entry for h in head_ids):
                        futures.append(executor.submit(save_slice_figure, slice_name, entry, head_ids, out_dir))
                for fut in as_completed(futures):
                    try:
                        fut.result()
                        saved += 1
                    except Exception as exc:
                        print(f"[warn] figure save failed: {exc}")
                    progress.advance(task_id, advance=1)
        else:
            for slice_name in names:
                entry = entries[slice_name]
                if "input" in entry and "target" in entry and not any(f"head{h}" not in entry for h in head_ids):
                    save_slice_figure(slice_name, entry, head_ids, out_dir)
                    saved += 1
                progress.advance(task_id, advance=1)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fusion-trained RegGAN model on a dataset.")
    parser.add_argument("--config", required=True, help="Path to evaluation config YAML.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to generator weights (e.g., save_root/netG_A2B.pth).",
    )
    parser.add_argument(
        "--reg-weights",
        default=None,
        help="Optional registration weights (R_A.pth).",
    )
    parser.add_argument("--device", default=None, help="Torch device string (default: cuda if available).")
    parser.add_argument("--compute-fid", action="store_true", help="Compute FID for each head and ensemble.")
    parser.add_argument("--fid-feature", type=int, default=64, help="Feature dimension for FID (default: 64).")
    parser.add_argument("--compute-lpips", action="store_true", help="Compute LPIPS for each head and ensemble.")
    parser.add_argument(
        "--lpips-net",
        default="vgg",
        choices=("alex", "vgg", "squeeze"),
        help="Backbone network for LPIPS (default: vgg).",
    )
    parser.add_argument("--output", default=None, help="Optional path to write JSON summary.")
    parser.add_argument(
        "--fig-dir",
        default=None,
        help="Optional directory to write per-slice visualization panels.",
    )
    parser.add_argument(
        "--fig-workers",
        type=int,
        default=0,
        help="Number of threads to save figures in parallel (default: 0 = single-threaded).",
    )
    parser.add_argument(
        "--max-figs",
        type=int,
        default=0,
        help="Optional cap on number of figures to save (0 = save all).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override dataset root (defaults to test/val/dataroot from config).")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    config["_fid_feature"] = int(args.fid_feature)
    config["_lpips_net"] = str(args.lpips_net)
    dataset_root = resolve_dataset_root(config, args.data_root)
    loader = build_eval_loader(config, dataset_root)
    weights_path = Path(args.weights).expanduser().resolve()
    reg_path = Path(args.reg_weights).expanduser().resolve() if args.reg_weights else None
    metrics_use_rd = bool(
        config.get(
            "test_metrics_use_rd",
            config.get("val_metrics_use_rd", config.get("metrics_use_rd", False)),
        )
    )

    print(f"[info] Evaluating fusion checkpoint {weights_path} ...")
    head_results, viz_entries, ensemble_acc, ensemble_fid = evaluate_fusion_model(
        config=config,
        loader=loader,
        device=device,
        generator_path=weights_path,
        reg_path=reg_path,
        compute_fid=bool(args.compute_fid),
        compute_lpips=bool(args.compute_lpips),
        metrics_use_rd=metrics_use_rd,
    )

    head_ids = sorted(head_results.keys())
    summary: Dict[str, Dict] = {"heads": []}
    print("\nPer-head metrics:")
    aggregates: Dict[str, float] = {"mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    extra_keys: Dict[str, float] = {}
    for head in head_ids:
        stats = head_results[head]
        head_summary = {
            "head": int(head),
            "mae": stats["mae"],
            "psnr": stats["psnr"],
            "ssim": stats["ssim"],
        }
        line = (
            f"  Head {head}: MAE={stats['mae']:.6f}, PSNR={stats['psnr']:.4f}, SSIM={stats['ssim']:.4f}"
        )
        if "fid" in stats:
            head_summary["fid"] = stats["fid"]
            line += f", FID={stats['fid']:.4f}"
            extra_keys.setdefault("fid", 0.0)
        if "lpips" in stats:
            head_summary["lpips"] = stats["lpips"]
            line += f", LPIPS={stats['lpips']:.6f}"
            extra_keys.setdefault("lpips", 0.0)
        summary["heads"].append(head_summary)
        print(line)
        aggregates["mae"] += stats["mae"]
        aggregates["psnr"] += stats["psnr"]
        aggregates["ssim"] += stats["ssim"]
        if "fid" in stats:
            extra_keys["fid"] += stats["fid"]
        if "lpips" in stats:
            extra_keys["lpips"] += stats["lpips"]

    if head_ids:
        inv = 1.0 / len(head_ids)
        avg_line = (
            f"\nPer-head mean: MAE={aggregates['mae'] * inv:.6f}, PSNR={aggregates['psnr'] * inv:.4f},"
            f" SSIM={aggregates['ssim'] * inv:.4f}"
        )
        if "fid" in extra_keys:
            avg_line += f", FID={extra_keys['fid'] * inv:.4f}"
        if "lpips" in extra_keys:
            avg_line += f", LPIPS={extra_keys['lpips'] * inv:.6f}"
        print(avg_line)
        summary["per_head_mean"] = {
            "mae": aggregates["mae"] * inv,
            "psnr": aggregates["psnr"] * inv,
            "ssim": aggregates["ssim"] * inv,
        }
        if "fid" in extra_keys:
            summary["per_head_mean"]["fid"] = extra_keys["fid"] * inv
        if "lpips" in extra_keys:
            summary["per_head_mean"]["lpips"] = extra_keys["lpips"] * inv

    ensemble_metrics = compute_ensemble_metrics(
        store=ensemble_acc,
        metrics_use_rd=metrics_use_rd,
        device=device,
        compute_fid=False,
        compute_lpips=bool(args.compute_lpips),
        lpips_net=str(args.lpips_net),
        precomputed_fid=ensemble_fid if bool(args.compute_fid) else None,
    )
    summary["ensemble"] = ensemble_metrics
    line = (
        f"\nAveraged prediction: MAE={ensemble_metrics['mae']:.6f}, PSNR={ensemble_metrics['psnr']:.4f},"
        f" SSIM={ensemble_metrics['ssim']:.4f}"
    )
    if "fid" in ensemble_metrics:
        line += f", FID={ensemble_metrics['fid']:.4f}"
    if "lpips" in ensemble_metrics:
        line += f", LPIPS={ensemble_metrics['lpips']:.6f}"
    print(line)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print(f"\nWrote summary to {out_path}")

    if args.fig_dir:
        fig_dir = Path(args.fig_dir).expanduser().resolve()
        saved = save_visualizations(
            viz_entries,
            head_ids,
            fig_dir,
            max_figs=int(args.max_figs),
            workers=int(args.fig_workers),
        )
        print(f"\nSaved {saved} visualizations to {fig_dir}")



if __name__ == "__main__":
    main()
