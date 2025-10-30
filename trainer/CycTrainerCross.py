#!/usr/bin/python3
"""
Staged, parallel cross-fold training for CycleGAN/RegGAN.

Each fold trains on its own GPU. Training proceeds in stages (default 5 epochs):
    train → IF/OOF inference → confidence-weight update → barrier → next stage

After every stage we persist:
    - checkpoints/checkpoint_stageXX.pth  (model + optimizers)
    - stage_XX/residuals/{if,oof}/maps/*.npy
    - stage_XX/weights/*.npy              (per-slice pixel weights)
    - stage_XX/metadata.json              (stage bookkeeping)

Weights from stage S feed into stage S+1 via `rd_mode='weights'`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import wandb

from .CycTrainer import Cyc_Trainer
from .datasets import ValDataset
from .utils import (
    Resize,
    ResizeKeepRatioPad,
    ToTensor,
    compose_slice_name,
    compute_mae,
    compute_psnr,
    compute_ssim,
)
from .confidence_utils import compute_weight_map


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _expand(path_like: Union[str, os.PathLike]) -> Path:
    return Path(path_like).expanduser().resolve()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_eval_transforms(config: Dict) -> List:
    mode = str(config.get("resize_mode", "resize")).lower()
    if mode not in ("resize", "keepratio"):
        mode = "resize"
    if mode == "keepratio":
        last_tf = ResizeKeepRatioPad(size_tuple=(config["size"], config["size"]), fill=-1)
    else:
        last_tf = Resize(size_tuple=(config["size"], config["size"]))
    return [ToTensor(), last_tf]


def _build_eval_loader(config: Dict, root_dir: Path) -> DataLoader:
    transforms_ = _build_eval_transforms(config)
    dataset = ValDataset(
        str(root_dir),
        transforms_=transforms_,
        unaligned=False,
        rd_input_type=None,
        rd_mask_dir="",
        rd_weights_dir="",
        rd_w_min=0.0,
        cache_mode=config.get("cache_mode", "mmap"),
        rd_cache_weights=config.get("rd_cache_weights", False),
    )
    n_workers = int(config.get("n_cpu", 0))
    kwargs = dict(
        batch_size=config["batchSize"],
        shuffle=False,
        num_workers=n_workers,
        pin_memory=bool(config.get("cuda", False)),
        timeout=int(config.get("dataloader_timeout", 0)),
    )
    if n_workers > 0:
        kwargs.update(
            persistent_workers=bool(config.get("persistent_workers", True)),
            prefetch_factor=max(1, int(config.get("prefetch_factor", 2))),
        )
    return DataLoader(dataset, **kwargs)


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _to_image(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[0] == 1:
            return a[0]
        if a.shape[-1] == 1:
            return a[..., 0]
    return np.squeeze(a)


def _collect_paths(directory: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    if not directory.is_dir():
        return mapping
    for path in directory.glob("*.npy"):
        mapping.setdefault(path.stem, []).append(path)
    return mapping

def _export_residuals(
    trainer: Cyc_Trainer,
    loader: DataLoader,
    out_root: Path,
    tag: str,
) -> List[Dict]:
    device = trainer.device if hasattr(trainer, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = trainer.netG_A2B
    use_reg = bool(trainer.config.get("eval_with_registration", False)) and hasattr(trainer, "R_A") and hasattr(trainer, "spatial_transform")
    reg_net = trainer.R_A if use_reg else None
    spatial_transform = trainer.spatial_transform if use_reg else None
    generator_was_train = generator.training
    generator.eval()
    if reg_net is not None:
        reg_was_train = reg_net.training
        reg_net.eval()
    else:
        reg_was_train = None
    maps_dir = _ensure_dir(out_root / "maps")
    records: List[Dict] = []
    torch.set_grad_enabled(False)
    for batch_idx, batch in enumerate(loader):
        real_A = batch["A"].to(device, non_blocking=True)
        real_Bt = batch["B"].to(device, non_blocking=True)
        fake_Bt = generator(real_A)
        fake_eval = fake_Bt
        if use_reg and reg_net is not None and spatial_transform is not None:
            Trans = reg_net(fake_Bt, real_Bt)
            fake_eval = spatial_transform(fake_Bt, Trans)
        real_np = _tensor_to_numpy(real_Bt)
        fake_np = _tensor_to_numpy(fake_eval)
        rd_weight = batch.get("rd_weight", None)
        if isinstance(rd_weight, torch.Tensor):
            rd_weight = rd_weight.detach().cpu().numpy()
        B = real_np.shape[0]
        for b in range(B):
            pred = _to_image(fake_np[b])
            gt = _to_image(real_np[b])
            body_mask = (gt != -1).astype(np.float32)
            weight = None
            if rd_weight is not None:
                w2d = _to_image(rd_weight[b])
                weight = np.clip(w2d, 0.0, 1.0) * body_mask
            slice_name = compose_slice_name(batch, batch_idx, b)
            residual = (pred - gt).astype(np.float32)
            np.save(maps_dir / f"{slice_name}.npy", residual)
            mae = compute_mae(pred, gt, mask=weight)
            psnr = compute_psnr(pred, gt, mask=weight)
            ssim = compute_ssim(pred, gt, mask=weight)
            pid = None
            sid = None
            pid_list = batch.get("patient_id", None)
            sid_list = batch.get("slice_id", None)
            if isinstance(pid_list, (list, tuple)) and b < len(pid_list):
                pid = pid_list[b]
            elif isinstance(pid_list, str):
                pid = pid_list
            if isinstance(sid_list, (list, tuple)) and b < len(sid_list):
                sid = sid_list[b]
            elif isinstance(sid_list, str):
                sid = sid_list
            records.append(
                {
                    "slice": slice_name,
                    "patient_id": pid,
                    "slice_id": sid,
                    "mae": mae,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )
    torch.set_grad_enabled(True)
    if generator_was_train:
        generator.train()
    if reg_was_train:
        reg_net.train()
    csv_path = out_root / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("slice,patient_id,slice_id,mae,psnr,ssim\n")
        for row in records:
            fh.write(f"{row['slice']},{row.get('patient_id','')},{row.get('slice_id','')},{row['mae']:.6f},{row['psnr']:.6f},{row['ssim']:.6f}\n")
    return records


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldPlan:
    fold_id: int
    train_dir: Path
    val_dir: Path
    save_root: Path
    device_id: int


@dataclass
class StageRuntime:
    stage_epochs: int
    total_epochs: int
    start_epoch: int
    confidence_cfg: Dict[str, float]

    @property
    def stage_count(self) -> int:
        remaining = max(0, self.total_epochs - self.start_epoch)
        return math.ceil(remaining / self.stage_epochs) if self.stage_epochs > 0 else 0


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class CycTrainerCross:
    def __init__(self, config: Dict):
        self.base_config = copy.deepcopy(config)
        cv_root = config.get("cv_root")
        save_root = config.get("save_root")
        if not cv_root or not save_root:
            raise KeyError("cv_root and save_root must be specified for cross training.")
        self.cv_root = _expand(cv_root)
        self.save_root = _ensure_dir(_expand(save_root))
        if not self.cv_root.is_dir():
            raise FileNotFoundError(f"cv_root '{self.cv_root}' does not exist.")
        requested = config.get("fold_ids")
        if requested is not None:
            requested = sorted({int(fid) for fid in requested})
        fold_dirs = sorted(p for p in self.cv_root.glob("fold_*") if p.is_dir())
        if not fold_dirs:
            raise RuntimeError(f"No fold_* directories found under {self.cv_root}.")
        device_ids = config.get("cv_gpus", None)
        if device_ids is None:
            device_ids = list(range(len(fold_dirs)))
        if len(device_ids) < len(fold_dirs):
            raise ValueError("Not enough GPU ids provided for the available folds.")
        self.fold_plans: List[FoldPlan] = []
        for idx, fold_dir in enumerate(fold_dirs):
            try:
                fold_id = int(fold_dir.name.split("_")[-1])
            except ValueError:
                continue
            if requested and fold_id not in requested:
                continue
            train_dir = fold_dir / "train"
            val_dir = fold_dir / "val"
            if not train_dir.is_dir() or not val_dir.is_dir():
                raise RuntimeError(f"Fold {fold_dir} missing train/val subdirectories.")
            fold_save_root = _ensure_dir(self.save_root / f"fold_{fold_id:02d}")
            device_id = int(device_ids[len(self.fold_plans) % len(device_ids)])
            self.fold_plans.append(
                FoldPlan(
                    fold_id=fold_id,
                    train_dir=train_dir,
                    val_dir=val_dir,
                    save_root=fold_save_root,
                    device_id=device_id,
                )
            )
        if not self.fold_plans:
            raise RuntimeError("No folds selected for training.")
        stage_epochs = int(config.get("stage_epochs", 5))
        if stage_epochs <= 0:
            raise ValueError("stage_epochs must be positive.")
        start_epoch = int(config.get("epoch", 0))
        total_epochs = int(config.get("n_epochs", 0))
        self.stage_runtime = StageRuntime(
            stage_epochs=stage_epochs,
            total_epochs=total_epochs,
            start_epoch=start_epoch,
            confidence_cfg={
                "thr_low": float(config.get("confidence_thr_low", 0.05)),
                "thr_high": float(config.get("confidence_thr_high", 0.15)),
                "gap_max": float(config.get("confidence_gap_max", 0.05)),
                "w_min": float(config.get("confidence_w_min", 0.05)),
                "lambda_if": float(config.get("confidence_lambda_if", 1.0)),
                "lambda_oof": float(config.get("confidence_lambda_oof", 2.0)),
                "lambda_gap": float(config.get("confidence_lambda_gap", 1.0)),
            },
        )

    def train(self):
        if self.stage_runtime.stage_count == 0:
            print("[cross] nothing to train (stage_count=0).")
            return
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(parties=len(self.fold_plans))
        procs: List[mp.Process] = []
        for rank, plan in enumerate(self.fold_plans):
            args = (
                rank,
                plan,
                self.fold_plans,
                copy.deepcopy(self.base_config),
                self.stage_runtime,
                barrier,
            )
            proc = ctx.Process(target=_fold_worker_entry, args=args, daemon=False)
            proc.start()
            procs.append(proc)
        errors = []
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                errors.append(proc.exitcode)
        if errors:
            raise RuntimeError(f"[cross] worker(s) exited with codes: {errors}")


# ---------------------------------------------------------------------------
# Worker routine
# ---------------------------------------------------------------------------


def _fold_worker_entry(
    rank: int,
    plan: FoldPlan,
    all_plans: List[FoldPlan],
    base_config: Dict,
    stage_runtime: StageRuntime,
    barrier: mp.Barrier,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan.device_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    fold_config = copy.deepcopy(base_config)
    fold_config["dataroot"] = str(plan.train_dir)
    fold_config["val_dataroot"] = str(plan.val_dir)
    fold_config["save_root"] = str(plan.save_root)
    fold_config["run_tag"] = f"{fold_config.get('name', 'CycleGan')}-fold{plan.fold_id:02d}"
    fold_config["ddp"] = False
    fold_config["epoch"] = int(fold_config.get("epoch", 0))
    fold_config["disable_progress"] = bool(rank != 0)
    fold_config["fold_id"] = plan.fold_id
    base_run_name = fold_config.get("wandb_run_name")
    if base_run_name:
        fold_config["wandb_run_name"] = f"{base_run_name}-fold{plan.fold_id:02d}"
    else:
        fold_config["wandb_run_name"] = f"{fold_config.get('name', 'CycleGan')}-fold{plan.fold_id:02d}"
    base_tags = fold_config.get("wandb_tags", [])
    if isinstance(base_tags, list):
        if f"fold{plan.fold_id:02d}" not in base_tags:
            fold_config["wandb_tags"] = base_tags + [f"fold{plan.fold_id:02d}"]
    _ensure_dir(plan.save_root)
    checkpoints_dir = _ensure_dir(plan.save_root / "checkpoints")
    trainer = Cyc_Trainer(fold_config)
    stage_log_samples = int(fold_config.get("stage_log_samples", 0))
    total_epochs = stage_runtime.total_epochs
    stage_epochs = stage_runtime.stage_epochs
    conf_cfg = stage_runtime.confidence_cfg
    train_eval_loader = _build_eval_loader(fold_config, plan.train_dir)
    val_eval_loader = _build_eval_loader(fold_config, plan.val_dir)
    # Resume existing stages if present
    stage_count = stage_runtime.stage_count
    if stage_count == 0:
        barrier.wait()
        return
    for stage_num in range(1, stage_count + 1):
        stage_dir = _ensure_dir(plan.save_root / f"stage_{stage_num:02d}")
        ckpt_prev = checkpoints_dir / f"stage_{stage_num - 1:02d}.pth"
        if stage_num > 1 and ckpt_prev.is_file():
            try:
                trainer.load_checkpoint(str(ckpt_prev))
            except Exception as exc:
                print(f"[fold {plan.fold_id}] failed to load checkpoint {ckpt_prev}: {exc}")
        if stage_num > 1:
            prev_weights = plan.save_root / f"stage_{stage_num - 1:02d}" / "weights"
            if prev_weights.is_dir():
                trainer.set_rd_weights(str(prev_weights), conf_cfg["w_min"])
        ckpt_path = checkpoints_dir / f"stage_{stage_num:02d}.pth"
        meta_path = stage_dir / "metadata.json"
        stage_completed = meta_path.is_file() and ckpt_path.is_file()
        weights_dir = stage_dir / "weights"
        if stage_completed:
            try:
                trainer.load_checkpoint(str(ckpt_path))
            except Exception as exc:
                print(f"[fold {plan.fold_id}] warning: failed to reload stage {stage_num} checkpoint ({exc}), recomputing.")
                stage_completed = False
        stage_start_epoch = trainer.get_epoch()
        stage_target_epoch = min(total_epochs, stage_start_epoch + stage_epochs)
        epochs_to_run = stage_target_epoch - stage_start_epoch
        if not stage_completed:
            if epochs_to_run > 0:
                print(f"[fold {plan.fold_id}] Stage {stage_num}: epochs {stage_start_epoch}→{stage_target_epoch}", flush=True)
                trainer.train(epochs_to_run)
            else:
                print(f"[fold {plan.fold_id}] Stage {stage_num}: zero additional epochs (already at target).", flush=True)
            trainer.save_checkpoint(str(ckpt_path), extra={"stage": stage_num})
            residual_dir = _ensure_dir(stage_dir / "residuals")
            print(f"[fold {plan.fold_id}] Stage {stage_num}: exporting OOF residuals", flush=True)
            _export_residuals(trainer, val_eval_loader, _ensure_dir(residual_dir / "oof"), "oof")
            print(f"[fold {plan.fold_id}] Stage {stage_num}: exporting IF residuals", flush=True)
            _export_residuals(trainer, train_eval_loader, _ensure_dir(residual_dir / "if"), "if")
            metadata = {
                "stage": stage_num,
                "epoch_range": [stage_start_epoch, trainer.get_epoch()],
                "checkpoint": str(ckpt_path),
                "weights_dir": str(weights_dir),
            }
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
        barrier.wait()
        if rank == 0:
            print(f"[cross] Stage {stage_num}: aggregating confidence weights", flush=True)
            _aggregate_stage_weights(all_plans, stage_num, conf_cfg)
        barrier.wait()
        weights_dir = _ensure_dir(stage_dir / "weights")
        metadata = {
            "stage": stage_num,
            "epoch_range": [stage_start_epoch, trainer.get_epoch()],
            "checkpoint": str(ckpt_path),
            "weights_dir": str(weights_dir),
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        if stage_log_samples > 0 and wandb.run is not None:
            try:
                _log_stage_samples(weights_dir, stage_num, plan.fold_id, stage_log_samples)
            except Exception as exc:
                print(f"[fold {plan.fold_id}] Stage {stage_num}: stage sample log failed ({exc})", flush=True)
        if stage_target_epoch < total_epochs:
            if any(weights_dir.glob("*.npy")):
                trainer.set_rd_weights(str(weights_dir), conf_cfg["w_min"])
            else:
                print(f"[fold {plan.fold_id}] Stage {stage_num}: no weights produced; keeping previous configuration.", flush=True)
    barrier.wait()


def _aggregate_stage_weights(plans: List[FoldPlan], stage_num: int, cfg: Dict[str, float]):
    stage_dirs = {plan.fold_id: plan.save_root / f"stage_{stage_num:02d}" for plan in plans}
    stage_start_time = time.perf_counter()
    oof_maps_by_fold = {
        plan.fold_id: _collect_paths(stage_dirs[plan.fold_id] / "residuals" / "oof" / "maps")
        for plan in plans
    }
    if_maps_by_fold = {
        plan.fold_id: _collect_paths(stage_dirs[plan.fold_id] / "residuals" / "if" / "maps")
        for plan in plans
    }
    slice_owner: Dict[str, tuple[int, Path]] = {}
    for fid, mapping in oof_maps_by_fold.items():
        for slice_name, paths in mapping.items():
            if paths:
                slice_owner[slice_name] = (fid, paths[0])

    for plan in plans:
        stage_dir = stage_dirs[plan.fold_id]
        residual_dir = stage_dir / "residuals"
        if not residual_dir.exists():
            continue
        weights_dir = _ensure_dir(stage_dir / "weights")
        comp_dir = _ensure_dir(weights_dir / "components")
        for old in weights_dir.glob("*.npy"):
            try:
                old.unlink()
            except FileNotFoundError:
                pass
        for old in comp_dir.glob("*"):
            try:
                old.unlink()
            except FileNotFoundError:
                pass
        stats = []
        target_if_map = if_maps_by_fold.get(plan.fold_id, {})
        total_slices = len(target_if_map)
        if total_slices <= 0:
            print(f"[cross] Stage {stage_num}: fold {plan.fold_id} has no IF residuals; skipping", flush=True)
            summary = {
                "count": 0,
                "avg_weight": None,
                "cfg": cfg,
            }
            with (weights_dir / "summary.json").open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            continue
        progress_interval = max(1, total_slices // 10)
        for slice_name, path_list in target_if_map.items():
            if not path_list:
                continue
            r_if_stack = []
            for fid, fmap in if_maps_by_fold.items():
                paths = fmap.get(slice_name)
                if not paths:
                    continue
                for p in paths:
                    r_if_stack.append(np.abs(np.load(p)).astype(np.float32))
            if not r_if_stack:
                continue
            if slice_name in slice_owner:
                r_oof = np.abs(np.load(slice_owner[slice_name][1])).astype(np.float32)
            else:
                r_oof = r_if_stack[0]
            result = compute_weight_map(
                r_oof=r_oof,
                r_if_stack=r_if_stack,
                gap_max=cfg["gap_max"],
                thr_low=cfg["thr_low"],
                thr_high=cfg["thr_high"],
                w_min=cfg["w_min"],
                lam_if=cfg["lambda_if"],
                lam_oof=cfg["lambda_oof"],
                lam_gap=cfg["lambda_gap"],
            )
            np.save(weights_dir / f"{slice_name}.npy", result["weights"])
            np.savez_compressed(
                comp_dir / f"{slice_name}.npz",
                c_if=result["c_if"],
                c_oof=result["c_oof"],
                c_gap=result["c_gap"],
                gap=result["gap"],
                r_oof=r_oof,
                r_if_mean=result["r_if_mean"],
            )
            stats.append(float(result["weights"].mean()))
            if len(stats) % progress_interval == 0:
                print(
                    f"[cross] Stage {stage_num}: fold {plan.fold_id} processed {len(stats)}/{total_slices} slices",
                    flush=True,
                )
        summary = {
            "count": len(stats),
            "avg_weight": float(np.mean(stats)) if stats else None,
            "cfg": cfg,
        }
        with (weights_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(
            f"[cross] Stage {stage_num}: fold {plan.fold_id} weights → count={summary['count']} avg={summary['avg_weight']}"
        )
    elapsed = time.perf_counter() - stage_start_time
    print(f"[cross] Stage {stage_num}: weight aggregation took {elapsed:.1f} s", flush=True)


CrossCycleTrainer = CycTrainerCross


def _log_stage_samples(weights_dir: Path, stage_num: int, fold_id: int, sample_count: int):
    weight_paths = sorted(weights_dir.glob("*.npy"))
    if not weight_paths:
        return
    comp_dir = weights_dir / "components"
    samples = weight_paths if len(weight_paths) <= sample_count else random.sample(weight_paths, sample_count)

    def _to_uint8(arr, diverging=False):
        a = np.asarray(arr)
        a = np.squeeze(a)
        if a.ndim != 2:
            raise ValueError("Expected 2D array for visualization")
        if diverging:
            vmax = np.max(np.abs(a))
            if vmax > 1e-6:
                norm = (a / (2 * vmax)) + 0.5
            else:
                norm = np.full_like(a, 0.5, dtype=np.float32)
        else:
            amin, amax = float(np.min(a)), float(np.max(a))
            if amax > amin:
                norm = (a - amin) / (amax - amin + 1e-6)
            else:
                norm = np.zeros_like(a, dtype=np.float32)
        return (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)

    table_columns = ["id", "weight", "gap", "residual_if", "residual_oof"]
    table_data = []
    for path in samples:
        stem = path.stem
        comp_path = comp_dir / f"{stem}.npz"
        try:
            comp = np.load(comp_path)
        except FileNotFoundError:
            continue
        weight_map = np.load(path)
        gap_map = comp.get("gap")
        residual_if = comp.get("r_if_mean")
        residual_oof = comp.get("r_oof")
        images = [
            wandb.Image(_to_uint8(weight_map), caption=f"{stem} weight"),
            wandb.Image(_to_uint8(gap_map, diverging=True), caption=f"{stem} gap") if gap_map is not None else None,
            wandb.Image(_to_uint8(np.abs(residual_if)), caption=f"{stem} |IF|") if residual_if is not None else None,
            wandb.Image(_to_uint8(np.abs(residual_oof)), caption=f"{stem} |OOF|") if residual_oof is not None else None,
        ]
        table_row = [stem]
        for img in images:
            table_row.append(img)
        table_data.append(table_row)

    if not table_data:
        return

    try:
        table = wandb.Table(columns=table_columns, data=table_data)
        wandb.log({
            f"stage/weights_fold{fold_id:02d}_ep{stage_num:02d}": table,
            "stage": stage_num,
        })
    except Exception:
        pass
