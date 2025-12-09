#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
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
)
from models.CycleGan import Generator

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except ImportError:
    LearnedPerceptualImagePatchSimilarity = None


def load_config(path: str) -> Dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def resolve_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_fold_dirs(base_config: Dict, cli_paths: Optional[List[str]]) -> List[Path]:
    if cli_paths:
        return [Path(p).expanduser().resolve() for p in cli_paths]
    save_root = base_config.get("save_root")
    if not save_root:
        raise ValueError("save_root missing; pass --fold-root.")
    root_path = Path(save_root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"save_root '{root_path}' not found.")
    fold_dirs = sorted(p for p in root_path.glob("fold_*") if p.is_dir())
    if not fold_dirs:
        raise RuntimeError(f"No fold_* directories found under {root_path}.")
    return fold_dirs


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


def build_eval_loader(config: Dict) -> DataLoader:
    mode = str(config.get("resize_mode", "resize")).lower()
    if mode not in ("resize", "keepratio"):
        mode = "resize"
    if mode == "keepratio":
        last_tf = ResizeKeepRatioPad(size_tuple=(config["size"], config["size"]), fill=-1)
    else:
        last_tf = Resize(size_tuple=(config["size"], config["size"]))
    transforms = [ToTensor(), last_tf]
    rd_mode = resolve_rd_mode(config)
    dataset = ValDataset(
        root=config["val_dataroot"],
        transforms_=transforms,
        unaligned=False,
        rd_input_type=rd_mode,
        rd_mask_dir=config.get("rd_mask_dir", ""),
        rd_weights_dir=config.get("rd_weights_dir", ""),
        rd_w_min=float(config.get("rd_w_min", 0.0)),
        cache_mode=config.get("cache_mode", "mmap"),
        rd_cache_weights=bool(config.get("rd_cache_weights", False)),
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


def load_generator(
    fold_dir: Path,
    config: Dict,
    device: torch.device,
    weight_name: str,
) -> Generator:
    path = fold_dir / weight_name
    if not path.is_file():
        raise FileNotFoundError(f"Generator weights not found: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "netG_A2B" in state:
        state = state["netG_A2B"]
    preferred_mode = str(config.get("generator_upsample_mode", "resize")).lower()
    tried_modes = []
    last_err: Optional[Exception] = None
    for mode in ([preferred_mode] + (["deconv"] if preferred_mode != "deconv" else [])):
        tried_modes.append(mode)
        net = Generator(config["input_nc"], config["output_nc"], upsample_mode=mode).to(device)
        try:
            net.load_state_dict(state)
            if mode != preferred_mode:
                print(
                    f"[warn] Loaded generator '{weight_name}' for {fold_dir.name} using legacy upsample_mode='{mode}'."
                )
            net.eval()
            return net
        except RuntimeError as err:
            last_err = err
    tried = ", ".join(tried_modes)
    raise RuntimeError(
        f"Failed to load generator weights from {path} with upsample modes [{tried}]."
    ) from last_err


def _resolve_reg_channels(config: Dict) -> Tuple[int, int]:
    fake_nc = int(config.get("reg_fake_nc", config.get("output_nc", config.get("input_nc", 1))))
    real_nc = int(config.get("reg_real_nc", config.get("output_nc", config.get("input_nc", 1))))
    return fake_nc, real_nc


def load_registration(
    fold_dir: Path,
    config: Dict,
    device: torch.device,
    weight_name: str,
) -> Tuple[Optional[Reg], Optional[Transformer_2D]]:
    if not (config.get("regist") and config.get("eval_with_registration")):
        return None, None
    path = fold_dir / weight_name
    if not path.is_file():
        print(f"[warn] Registration weights missing at {path}; skipping registration.")
        return None, None
    fake_nc, real_nc = _resolve_reg_channels(config)
    reg_net = Reg(config["size"], config["size"], fake_nc, real_nc).to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "R_A" in state:
        state = state["R_A"]
    reg_net.load_state_dict(state)
    reg_net.eval()
    transformer = Transformer_2D().to(device)
    return reg_net, transformer


def squeeze_hw(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    out = np.squeeze(out)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {out.shape}")
    return out.astype(np.float32, copy=False)


def build_keep_mask(
    real: np.ndarray,
    rd_weight_batch: Optional[np.ndarray],
    index: int,
) -> np.ndarray:
    body = (real != -1).astype(np.float32)
    if rd_weight_batch is None:
        return body
    weight = squeeze_hw(rd_weight_batch[index])
    if weight.shape != real.shape:
        weight = cv2.resize(weight.astype(np.float32), (real.shape[1], real.shape[0]), interpolation=cv2.INTER_NEAREST)
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


def run_inference(
    fold_dir: Path,
    loader: DataLoader,
    config: Dict,
    device: torch.device,
    generator_name: str,
    reg_name: str,
    ensemble: Optional[EnsembleAccumulator],
    metrics_use_rd: bool,
    viz_store: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict:
    generator = load_generator(fold_dir, config, device, generator_name)
    reg_net, transformer = load_registration(fold_dir, config, device, reg_name)
    result = {
        "fold": fold_dir.name,
        "mae_sum": 0.0,
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "count": 0,
        "fid": None,
        "lpips_sum": 0.0,
        "lpips_count": 0.0,
    }
    fid_metric = None
    lpips_metric = None
    if config.get("_compute_fid"):
        if FrechetInceptionDistance is None:
            raise RuntimeError("FrechetInceptionDistance not available; install torchmetrics[image].")
        fid_metric = FrechetInceptionDistance(feature=config.get("_fid_feature", 64)).to(device)
    if config.get("_compute_lpips"):
        if LearnedPerceptualImagePatchSimilarity is None:
            raise RuntimeError("LPIPS metric not available; install torchmetrics[image].")
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type=config.get("_lpips_net", "vgg"),
            normalize=True,
        ).to(device)
    torch.set_grad_enabled(False)
    try:
        total_slices = len(loader.dataset)  # type: ignore[attr-defined]
    except Exception:
        total_slices = None
    progress = Progress(
        TextColumn(f"[bold blue]{fold_dir.name}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        task_id = progress.add_task(
            "eval",
            total=total_slices if (isinstance(total_slices, int) and total_slices > 0) else None,
        )
        for batch_idx, batch in enumerate(loader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            fake_B = generator(real_A)
            fake_eval = fake_B
            if reg_net is not None and transformer is not None:
                flow = reg_net(fake_B, real_B)
                fake_eval = transformer(fake_B, flow)
            real_np = real_B.detach().cpu().numpy()
            fake_np = fake_eval.detach().cpu().numpy()
            rd_weight = batch.get("rd_weight")
            if isinstance(rd_weight, torch.Tensor):
                rd_weight_np = rd_weight.detach().cpu().numpy()
            else:
                rd_weight_np = None
            batch_count = real_np.shape[0]
            if fid_metric is not None or lpips_metric is not None:
                real_for_metrics = prepare_images_for_metrics(real_B)
                fake_for_metrics = prepare_images_for_metrics(fake_eval)
                real_uint8 = torch.clamp(real_for_metrics * 255.0, 0.0, 255.0).to(torch.uint8)
                fake_uint8 = torch.clamp(fake_for_metrics * 255.0, 0.0, 255.0).to(torch.uint8)
                if fid_metric is not None:
                    fid_metric.update(real_uint8, real=True)
                    fid_metric.update(fake_uint8, real=False)
                if lpips_metric is not None:
                    lpips_val = lpips_metric(fake_for_metrics, real_for_metrics)
                    result["lpips_sum"] += float(lpips_val.sum().item())
                    result["lpips_count"] += float(lpips_val.numel())
            for b in range(batch_count):
                slice_name = compose_slice_name(batch, batch_idx, b)
                real_2d = squeeze_hw(real_np[b])
                fake_2d = squeeze_hw(fake_np[b])
                if viz_store is not None:
                    entry = viz_store.setdefault(slice_name, {})
                    entry.setdefault("input", squeeze_hw(batch["A"].detach().cpu().numpy()[b]))
                    entry.setdefault("target", real_2d)
                    entry[fold_dir.name] = fake_2d
                keep_mask = None
                if metrics_use_rd:
                    keep_mask = build_keep_mask(real_2d, rd_weight_np, b)
                mae = compute_mae(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                psnr = compute_psnr(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                ssim = compute_ssim(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                result["mae_sum"] += mae
                result["psnr_sum"] += psnr
                result["ssim_sum"] += ssim
                result["count"] += 1
                if ensemble is not None:
                    meta = {
                        "patient_id": _safe_index(batch.get("patient_id"), b),
                        "slice_id": _safe_index(batch.get("slice_id"), b),
                    }
                    ensemble.add(slice_name, fake_2d, real_2d, keep_mask if metrics_use_rd else None, meta)
            progress.advance(task_id, advance=batch_count)
    if fid_metric is not None:
        result["fid"] = float(fid_metric.compute().item())
    if result["lpips_count"] > 0:
        result["lpips"] = result["lpips_sum"] / result["lpips_count"]
    else:
        result["lpips"] = None
    result.pop("lpips_sum", None)
    result.pop("lpips_count", None)
    return result


def _safe_index(seq: Optional[Iterable], idx: int):
    if seq is None:
        return None
    if isinstance(seq, (list, tuple)):
        return seq[idx] if idx < len(seq) else None
    return seq


def summarise_fold_metrics(fold_results: List[Dict]) -> Dict[str, float]:
    summary = {"folds": []}
    for fr in fold_results:
        if fr["count"] == 0:
            raise RuntimeError(f"Fold {fr['fold']} produced no samples.")
        fold_metrics = {
            "fold": fr["fold"],
            "mae": fr["mae_sum"] / fr["count"],
            "psnr": fr["psnr_sum"] / fr["count"],
            "ssim": fr["ssim_sum"] / fr["count"],
        }
        if fr.get("fid") is not None:
            fold_metrics["fid"] = fr["fid"]
        if fr.get("lpips") is not None:
            fold_metrics["lpips"] = fr["lpips"]
        summary["folds"].append(fold_metrics)
    summary["average"] = {
        metric: float(np.mean([f[metric] for f in summary["folds"]]))
        for metric in ("mae", "psnr", "ssim")
    }
    if any("fid" in f for f in summary["folds"]):
        summary["average"]["fid"] = float(
            np.mean([f["fid"] for f in summary["folds"] if "fid" in f])
        )
    if any("lpips" in f for f in summary["folds"]):
        summary["average"]["lpips"] = float(
            np.mean([f["lpips"] for f in summary["folds"] if "lpips" in f])
        )
    return summary


def compute_ensemble_metrics(
    store: EnsembleAccumulator,
    metrics_use_rd: bool,
    device: torch.device,
    compute_fid: bool = False,
    compute_lpips: bool = False,
    fid_feature: int = 64,
    lpips_net: str = "vgg",
) -> Dict[str, float]:
    entries = store.finalize()
    mae_sum = psnr_sum = ssim_sum = 0.0
    count = 0
    fid_metric = None
    lpips_metric = None
    lpips_sum = lpips_count = 0.0
    if compute_fid:
        if FrechetInceptionDistance is None:
            raise RuntimeError("FrechetInceptionDistance not available; install torchmetrics[image].")
        fid_metric = FrechetInceptionDistance(feature=fid_feature).to(device)
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
        if fid_metric is not None or lpips_metric is not None:
            fake_tensor = torch.from_numpy(avg_pred).unsqueeze(0).unsqueeze(0).to(device)
            real_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device)
            fake_imgs = prepare_images_for_metrics(fake_tensor)
            real_imgs = prepare_images_for_metrics(real_tensor)
            if fid_metric is not None:
                real_uint8 = torch.clamp(real_imgs * 255.0, 0.0, 255.0).to(torch.uint8)
                fake_uint8 = torch.clamp(fake_imgs * 255.0, 0.0, 255.0).to(torch.uint8)
                fid_metric.update(real_uint8, real=True)
                fid_metric.update(fake_uint8, real=False)
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
    if fid_metric is not None:
        metrics["fid"] = float(fid_metric.compute().item())
    if lpips_metric is not None and lpips_count > 0:
        metrics["lpips"] = lpips_sum / lpips_count
    return metrics


def _slice_to_display(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        a = np.squeeze(a)
    return a


def save_slice_figure(
    slice_name: str,
    entry: Dict[str, np.ndarray],
    fold_order: List[str],
    out_dir: Path,
) -> None:
    input_img = _slice_to_display(entry["input"])
    target_img = _slice_to_display(entry["target"])
    preds = []
    for name in fold_order:
        preds.append(_slice_to_display(entry[name]))
    avg_img = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)
    images = [input_img, target_img] + preds + [avg_img]
    titles = ["Input", "Target"] + [f"{name} pred" for name in fold_order] + ["Average"]
    vmin, vmax = -1.0, 1.0
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3), constrained_layout=True)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    out_path = out_dir / f"{slice_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_visualizations(
    entries: Dict[str, Dict[str, np.ndarray]],
    fold_order: List[str],
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for slice_name in sorted(entries.keys()):
        entry = entries[slice_name]
        missing = [name for name in fold_order if name not in entry]
        if missing:
            print(f"[warn] skipping {slice_name}: missing predictions from {missing}")
            continue
        if "input" not in entry or "target" not in entry:
            print(f"[warn] skipping {slice_name}: missing input/target reference")
            continue
        save_slice_figure(slice_name, entry, fold_order, out_dir)
        saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-trained model evaluation utility.")
    parser.add_argument("--config", required=True, help="Path to evaluation config YAML.")
    parser.add_argument(
        "--fold-root",
        action="append",
        dest="fold_roots",
        help="Fold directory containing weights (repeatable). Defaults to save_root/fold_*.",
    )
    parser.add_argument(
        "--mode",
        choices=("metrics", "ensemble", "both"),
        default="both",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--generator-name",
        default="netG_A2B.pth",
        help="Filename of generator weights inside each fold directory.",
    )
    parser.add_argument(
        "--reg-name",
        default="R_A.pth",
        help="Filename of registration weights inside each fold directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--compute-fid",
        action="store_true",
        help="Compute Frechet Inception Distance (requires torchmetrics[image]).",
    )
    parser.add_argument(
        "--fid-feature",
        type=int,
        default=64,
        help="Feature dimensionality for FID computation (default: 64).",
    )
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        help="Compute LPIPS (requires torchmetrics[image]).",
    )
    parser.add_argument(
        "--lpips-net",
        default="vgg",
        choices=("alex", "vgg", "squeeze"),
        help="Backbone network for LPIPS (default: vgg).",
    )
    parser.add_argument(
        "--fig-dir",
        default=None,
        help="Directory to save per-slice visualization panels (input/target/predictions/average).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    if int(config.get("mc_runs", 1)) != 1:
        raise NotImplementedError("MC inference (mc_runs>1) is not supported in this evaluator.")
    config["_compute_fid"] = bool(args.compute_fid)
    config["_fid_feature"] = int(args.fid_feature)
    config["_compute_lpips"] = bool(args.compute_lpips)
    config["_lpips_net"] = str(args.lpips_net)
    fold_dirs = parse_fold_dirs(config, args.fold_roots)
    fold_names = [fold_dir.name for fold_dir in fold_dirs]
    loader = build_eval_loader(config)
    metrics_use_rd = bool(
        config.get(
            "test_metrics_use_rd",
            config.get("val_metrics_use_rd", config.get("metrics_use_rd", False)),
        )
    )
    need_ensemble = args.mode in ("ensemble", "both")
    accumulator = EnsembleAccumulator(expected_models=len(fold_dirs)) if need_ensemble else None
    viz_entries: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    fig_dir: Optional[Path] = None
    if args.fig_dir:
        fig_dir = Path(args.fig_dir).expanduser().resolve()
        fig_dir.mkdir(parents=True, exist_ok=True)
        viz_entries = {}
    fold_results = []
    for fold_dir in fold_dirs:
        print(f"[info] Evaluating fold at {fold_dir} ...")
        res = run_inference(
            fold_dir=fold_dir,
            loader=loader,
            config=config,
            device=device,
            generator_name=args.generator_name,
            reg_name=args.reg_name,
            ensemble=accumulator,
            metrics_use_rd=metrics_use_rd,
            viz_store=viz_entries,
        )
        fold_results.append(res)
    summary: Dict[str, Dict] = {}
    if args.mode in ("metrics", "both"):
        metrics_summary = summarise_fold_metrics(fold_results)
        summary["per_fold_metrics"] = metrics_summary
        print("\nPer-fold metrics (averaged over slices):")
        for fold in metrics_summary["folds"]:
            extras = []
            if "fid" in fold:
                extras.append(f"FID={fold['fid']:.4f}")
            if "lpips" in fold:
                extras.append(f"LPIPS={fold['lpips']:.6f}")
            extra_txt = f", {' ,'.join(extras)}" if extras else ""
            print(
                f"  {fold['fold']}: MAE={fold['mae']:.6f}, "
                f"PSNR={fold['psnr']:.4f}, SSIM={fold['ssim']:.4f}{extra_txt}"
            )
        avg = metrics_summary["average"]
        extras_avg = []
        if "fid" in avg:
            extras_avg.append(f"FID={avg['fid']:.4f}")
        if "lpips" in avg:
            extras_avg.append(f"LPIPS={avg['lpips']:.6f}")
        extra_avg_txt = f", {' ,'.join(extras_avg)}" if extras_avg else ""
        print(
            f"Mean of fold metrics → MAE={avg['mae']:.6f}, "
            f"PSNR={avg['psnr']:.4f}, SSIM={avg['ssim']:.4f}{extra_avg_txt}"
        )
    if need_ensemble and accumulator is not None:
        ensemble_summary = compute_ensemble_metrics(
            accumulator,
            metrics_use_rd,
            device=device,
            compute_fid=bool(args.compute_fid),
            compute_lpips=bool(args.compute_lpips),
            fid_feature=int(args.fid_feature),
            lpips_net=str(args.lpips_net),
        )
        summary["ensemble_metrics"] = ensemble_summary
        extras_ens = []
        if "fid" in ensemble_summary:
            extras_ens.append(f"FID={ensemble_summary['fid']:.4f}")
        if "lpips" in ensemble_summary:
            extras_ens.append(f"LPIPS={ensemble_summary['lpips']:.6f}")
        extra_ens_txt = f", {' ,'.join(extras_ens)}" if extras_ens else ""
        print(
            f"\nEnsembled predictions ({ensemble_summary['slices']} slices): "
            f"MAE={ensemble_summary['mae']:.6f}, "
            f"PSNR={ensemble_summary['psnr']:.4f}, "
            f"SSIM={ensemble_summary['ssim']:.4f}{extra_ens_txt}"
        )
    if fig_dir is not None and viz_entries is not None:
        saved = save_visualizations(viz_entries, fold_names, fig_dir)
        print(f"\nSaved {saved} slice visualizations to {fig_dir}")
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print(f"\nWrote summary to {output_path}")


if __name__ == "__main__":
    main()
