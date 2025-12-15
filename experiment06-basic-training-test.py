#!/usr/bin/env python3

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
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
from models.CycleGan import Generator

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
    raise KeyError("Dataset root not specified; provide --data-root or set test_dataroot/val_dataroot/dataroot in config.")


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


def build_eval_loader(config: Dict, dataset_root: str) -> DataLoader:
    rd_mode = resolve_rd_mode(config)
    rd_input_type = rd_mode
    rd_mask_dir = config.get("rd_mask_dir", "")
    rd_weights_dir = config.get("rd_weights_dir", "")
    rd_w_min = float(config.get("rd_w_min", 0.0))
    cache_mode = str(config.get("cache_mode", "mmap"))
    rd_cache_weights = bool(config.get("rd_cache_weights", False))
    rd_fallback_mode = str(config.get("rd_fallback_mode", "body"))
    size = int(config.get("size", 256))
    resize_mode = str(config.get("resize_mode", "resize")).lower()
    if resize_mode == "keepratio":
        last_tf = ResizeKeepRatioPad(size_tuple=(size, size), fill=-1)
    else:
        last_tf = Resize(size_tuple=(size, size))
    transforms_ = [ToTensor(), last_tf]
    domain_a_dir = _resolve_dir_with_base(dataset_root, config.get("domain_a_dir", None)) or str(Path(dataset_root) / "A")
    domain_b_dir = _resolve_dir_with_base(dataset_root, config.get("domain_b_dir", None)) or str(Path(dataset_root) / "B")
    domain_a_channels = config.get("domain_a_channels", config.get("input_nc"))
    domain_b_channels = config.get("domain_b_channels", config.get("output_nc"))
    ds = ValDataset(
        root=dataset_root,
        transforms_=transforms_,
        unaligned=False,
        rd_input_type=rd_input_type,
        rd_mask_dir=rd_mask_dir,
        rd_weights_dir=rd_weights_dir,
        rd_w_min=rd_w_min,
        cache_mode=cache_mode,
        rd_cache_weights=rd_cache_weights,
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        domain_a_channels=domain_a_channels,
        domain_b_channels=domain_b_channels,
        rd_fallback_mode=rd_fallback_mode,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("batchSize", 1)),
        shuffle=False,
        num_workers=int(config.get("n_cpu", 0)),
        pin_memory=bool(config.get("cuda", True)),
    )
    return loader


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


def load_basic_generator(config: Dict, checkpoint: Path, device: torch.device) -> Generator:
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
    upsample_mode = str(config.get("generator_upsample_mode", "resize")).lower()
    net = Generator(config["input_nc"], config["output_nc"], upsample_mode=upsample_mode).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def _resolve_reg_channels(config: Dict) -> Tuple[int, int]:
    fake_nc = int(config.get("reg_fake_nc", config.get("output_nc", config.get("input_nc", 1))))
    real_nc = int(config.get("reg_real_nc", config.get("output_nc", config.get("input_nc", 1))))
    return fake_nc, real_nc


def load_registration(config: Dict, checkpoint: Optional[Path], device: torch.device) -> Tuple[Optional[Reg], Optional[Transformer_2D]]:
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


def squeeze_hw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return a


def build_keep_mask(real_2d: np.ndarray, rd_weight_batch: Optional[np.ndarray], batch_index: int) -> Optional[np.ndarray]:
    keep = (real_2d != -1).astype(np.float32)
    if rd_weight_batch is not None and rd_weight_batch.shape[0] > batch_index:
        w = rd_weight_batch[batch_index]
        if w is not None:
            w2d = squeeze_hw(w)
            if w2d.shape == real_2d.shape:
                keep = keep * np.clip(w2d, 0.0, 1.0)
    if np.sum(keep) <= 0:
        return None
    return keep


def _initialize_inception_extractor(device: torch.device):
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return model, preprocess


def _compute_inception_activations(images: torch.Tensor, model, preprocess, device: torch.device) -> np.ndarray:
    imgs = prepare_images_for_metrics(images)
    imgs = preprocess(imgs)
    with torch.no_grad():
        feats = model(imgs.to(device))
    return feats.detach().cpu().numpy()


def _calculate_fid_from_activations(fake_acts: np.ndarray, real_acts: np.ndarray) -> float:
    mu1, sigma1 = fake_acts.mean(axis=0), np.cov(fake_acts, rowvar=False)
    mu2, sigma2 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = covmean.real
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def save_slice_figure(slice_name: str, entry: Dict[str, np.ndarray], out_dir: Path):
    inp = entry.get("input")
    tgt = entry.get("target")
    pred = entry.get("pred")
    images = [to_uint8_image(inp), to_uint8_image(tgt), to_uint8_image(pred)]
    titles = ["Input", "Target", "Prediction"]
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


def save_visualizations(entries: Dict[str, Dict[str, np.ndarray]], out_dir: Path, max_figs: int = 0, workers: int = 0) -> int:
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
                    if "input" in entry and "target" in entry and "pred" in entry:
                        futures.append(executor.submit(save_slice_figure, slice_name, entry, out_dir))
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
                if "input" in entry and "target" in entry and "pred" in entry:
                    save_slice_figure(slice_name, entry, out_dir)
                    saved += 1
                progress.advance(task_id, advance=1)
    return saved


def evaluate_basic_model(
    config: Dict,
    loader: DataLoader,
    device: torch.device,
    generator_path: Path,
    reg_path: Optional[Path],
    compute_fid: bool,
    compute_lpips: bool,
    metrics_use_rd: bool,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, np.ndarray]]]:
    generator = load_basic_generator(config, generator_path, device)
    reg_net, transformer = load_registration(config, reg_path, device)
    results = {"mae_sum": 0.0, "psnr_sum": 0.0, "ssim_sum": 0.0, "count": 0}
    fid_model = None
    fid_preprocess = None
    fid_real_store: List[np.ndarray] = []
    fid_fake_store: List[np.ndarray] = []
    lpips_module = None
    if compute_fid:
        fid_model, fid_preprocess = _initialize_inception_extractor(device)
    if compute_lpips:
        if LearnedPerceptualImagePatchSimilarity is None:
            raise RuntimeError("LPIPS metric not available; install torchmetrics[image].")
        lpips_net = str(config.get("_lpips_net", "vgg"))
        lpips_module = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net, normalize=True).to(device)

    viz_entries: Dict[str, Dict[str, np.ndarray]] = {}

    torch.set_grad_enabled(False)
    try:
        total_slices = len(loader.dataset)  # type: ignore[attr-defined]
    except Exception:
        total_slices = None
    progress = Progress(
        TextColumn("[bold blue]basic[/]"),
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
            output = generator(real_A)
            if reg_net is not None and transformer is not None:
                flow = reg_net(output, real_B)
                output = transformer(output, flow)

            real_np = real_B.detach().cpu().numpy()
            rd_weight = batch.get("rd_weight")
            rd_weight_np = rd_weight.detach().cpu().numpy() if isinstance(rd_weight, torch.Tensor) else None

            # Metrics modules update
            if fid_model is not None and fid_preprocess is not None:
                real_feats = _compute_inception_activations(real_B, fid_model, fid_preprocess, device)
                fake_feats = _compute_inception_activations(output, fid_model, fid_preprocess, device)
                fid_real_store.append(real_feats)
                fid_fake_store.append(fake_feats)
            if lpips_module is not None:
                real_for_metrics = prepare_images_for_metrics(real_B)
                fake_imgs = prepare_images_for_metrics(output)
                lpips_val = lpips_module(fake_imgs, real_for_metrics)
                results.setdefault("lpips_sum", 0.0)
                results.setdefault("lpips_count", 0.0)
                results["lpips_sum"] += float(lpips_val.sum().item())
                results["lpips_count"] += float(lpips_val.numel())

            batch_count = real_np.shape[0]
            for b in range(batch_count):
                slice_name = compose_slice_name(batch, batch_idx, b)
                real_2d = squeeze_hw(real_np[b])
                keep_mask = build_keep_mask(real_2d, rd_weight_np, b) if metrics_use_rd else None

                if slice_name not in viz_entries:
                    viz_entries[slice_name] = {}
                    viz_entries[slice_name]["input"] = squeeze_hw(batch["A"].detach().cpu().numpy()[b])
                    viz_entries[slice_name]["target"] = real_2d

                fake_2d = squeeze_hw(output.detach().cpu().numpy()[b])
                mae = compute_mae(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                psnr = compute_psnr(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                ssim = compute_ssim(fake_2d, real_2d, mask=keep_mask if metrics_use_rd else None)
                results["mae_sum"] += mae
                results["psnr_sum"] += psnr
                results["ssim_sum"] += ssim
                results["count"] += 1
                viz_entries[slice_name]["pred"] = fake_2d

            progress.advance(task_id, advance=batch_count)

    if compute_fid and fid_real_store and fid_fake_store:
        real_acts = np.concatenate(fid_real_store, axis=0)
        fake_acts = np.concatenate(fid_fake_store, axis=0)
        results["fid"] = _calculate_fid_from_activations(fake_acts, real_acts)
    if results.get("lpips_count", 0.0) > 0:
        results["lpips"] = results["lpips_sum"] / results["lpips_count"]
    if results["count"] == 0:
        raise RuntimeError("No samples were processed.")
    results["mae"] = results["mae_sum"] / results["count"]
    results["psnr"] = results["psnr_sum"] / results["count"]
    results["ssim"] = results["ssim_sum"] / results["count"]
    return results, viz_entries


def main():
    parser = argparse.ArgumentParser(description="Evaluate basic (non-fusion) RegGAN model on a dataset.")
    parser.add_argument("--config", required=True, help="Path to evaluation config YAML.")
    parser.add_argument("--weights", required=True, help="Path to generator weights (e.g., save_root/netG_A2B.pth).")
    parser.add_argument("--reg-weights", default=None, help="Optional registration weights (R_A.pth).")
    parser.add_argument("--device", default=None, help="Torch device string (default: cuda if available).")
    parser.add_argument("--compute-fid", action="store_true", help="Compute FID.")
    parser.add_argument("--fid-feature", type=int, default=64, help="Feature dimension for FID (kept for parity).")
    parser.add_argument("--compute-lpips", action="store_true", help="Compute LPIPS.")
    parser.add_argument(
        "--lpips-net",
        default="vgg",
        choices=("alex", "vgg", "squeeze"),
        help="Backbone network for LPIPS (default: vgg).",
    )
    parser.add_argument("--output", default=None, help="Optional path to write JSON summary.")
    parser.add_argument("--fig-dir", default=None, help="Optional directory to write per-slice visualization panels.")
    parser.add_argument("--fig-workers", type=int, default=0, help="Threads to save figures in parallel (default: 0).")
    parser.add_argument("--max-figs", type=int, default=0, help="Cap on number of figures to save (0 = all).")
    parser.add_argument("--data-root", default=None, help="Override dataset root (defaults to test/val/dataroot).")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    config["_lpips_net"] = str(args.lpips_net)
    dataset_root = resolve_dataset_root(config, args.data_root)
    loader = build_eval_loader(config, dataset_root)
    weights_path = Path(args.weights).expanduser().resolve()
    reg_path = Path(args.reg_weights).expanduser().resolve() if args.reg_weights else None
    metrics_use_rd = bool(
        config.get("test_metrics_use_rd", config.get("val_metrics_use_rd", config.get("metrics_use_rd", False)))
    )

    print(f"[info] Evaluating basic checkpoint {weights_path} ...")
    results, viz_entries = evaluate_basic_model(
        config=config,
        loader=loader,
        device=device,
        generator_path=weights_path,
        reg_path=reg_path,
        compute_fid=bool(args.compute_fid),
        compute_lpips=bool(args.compute_lpips),
        metrics_use_rd=metrics_use_rd,
    )

    summary = {
        "mae": results["mae"],
        "psnr": results["psnr"],
        "ssim": results["ssim"],
    }
    line = f"\nMAE={results['mae']:.6f}, PSNR={results['psnr']:.4f}, SSIM={results['ssim']:.4f}"
    if "fid" in results:
        summary["fid"] = results["fid"]
        line += f", FID={results['fid']:.4f}"
    if "lpips" in results:
        summary["lpips"] = results["lpips"]
        line += f", LPIPS={results['lpips']:.6f}"
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
            fig_dir,
            max_figs=int(args.max_figs),
            workers=int(args.fig_workers),
        )
        print(f"\nSaved {saved} visualizations to {fig_dir}")


if __name__ == "__main__":
    main()
