#!/usr/bin/env python3
"""
Experiment 03 — Prepare K-fold splits with patient-level grouping.
-----------------------------------------------------------------
This utility builds cross-validation splits by patient ID and materialises
each fold as a set of symbolic links under a target directory. For every
fold ``f`` we create the layout::

    <output_root>/fold_<f+1>/train/<modality>/...
    <output_root>/fold_<f+1>/val/<modality>/...

The ``train`` split contains all patients except those assigned to fold ``f``;
``val`` contains the held-out patients. Symlinks always point back to the
original dataset, so storage overhead is negligible.

Key properties:
* Splits are computed at the PATIENT level (no slice-level leakage).
* Modalities (e.g. ``A``/``B``) are handled jointly so all channels stay in sync.
* Auto-discovery skips helper folders such as ``*-viz``; pass ``--modalities`` to
  override the list explicitly when needed.
* Patient IDs can be inferred either from per-patient directories or from
  filename stems (configurable via ``--id-regex``).

Example usage:

    python experiment03-1-cross-validation-data-split.py \
        --dataroot data/SynthRAD2023-Task1/train2D-foreground \
        --output-root data/SynthRAD2023-Task1/cv_folds \
        --folds 5 --seed 2025

After running, you can point each fold-specific config to the corresponding
``train``/``val`` directories.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set


# -----------------------------
# Argument parsing
# -----------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a training set into K folds by patient ID and materialise symlinked directories."
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        required=True,
        help="Root directory that contains modality sub-directories (e.g. A, B).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory where fold_<i> sub-folders will be created.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds to generate (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for patient shuffling (default: 2025).",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Explicit list of modality sub-directories to include (e.g. A B). "
            "If omitted we auto-detect immediate sub-directories under --dataroot."
        ),
    )
    parser.add_argument(
        "--id-regex",
        type=str,
        default=r"^(?P<id>.+?)(?:\.nii)?(?:[_-](?:z|slice|sl|s|frame)\d+)?$",
        help=(
            "Regex applied to file stems (without extension) when patient IDs cannot be inferred "
            "from directory names. The pattern must contain a named capture group 'id'."
        ),
    )
    parser.add_argument(
        "--include-ext",
        type=str,
        nargs="*",
        default=[".npy", ".png", ".jpg", ".jpeg", ".nii", ".nii.gz"],
        help="File extensions to consider during discovery (default: common medical image exports).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing --output-root by deleting it first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned fold assignment without writing symlinks.",
    )
    return parser


# -----------------------------
# Helpers
# -----------------------------


def _infer_modalities(dataroot: Path, explicit_modalities: Sequence[str] | None) -> List[str]:
    if explicit_modalities:
        modalities = list(dict.fromkeys(explicit_modalities))  # preserve order, drop dups
    else:
        ignore = re.compile(r"[-_]viz$", re.IGNORECASE)
        modalities = sorted(
            d.name
            for d in dataroot.iterdir()
            if d.is_dir() and not ignore.search(d.name) and not d.name.startswith('.')
        )
    if not modalities:
        raise RuntimeError(f"No modality directories found under {dataroot}.")
    missing = [m for m in modalities if not (dataroot / m).is_dir()]
    if missing:
        raise RuntimeError(f"Modalities not found under {dataroot}: {', '.join(missing)}")
    return modalities


def _gather_by_patient_from_subdirs(mod_dir: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for patient_dir in sorted(p for p in mod_dir.iterdir() if p.is_dir()):
        files = [f for f in patient_dir.rglob("*") if f.is_file()]
        if files:
            mapping[patient_dir.name] = sorted(files)
    return mapping


def _build_id_parser(pattern: str):
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --id-regex pattern: {exc}") from exc
    if "id" not in regex.groupindex:
        raise ValueError("--id-regex must define a named capture group 'id'.")
    return regex


def _infer_patient_from_filename(path: Path, regex) -> str:
    stem = path.stem
    match = regex.match(stem)
    if not match:
        raise ValueError(f"Could not extract patient ID from filename '{path.name}' using pattern '{regex.pattern}'.")
    patient_id = str(match.group("id")).strip()
    if not patient_id:
        raise ValueError(f"Empty patient ID detected for '{path.name}'.")
    return patient_id


def _gather_by_patient_from_files(mod_dir: Path, include_ext: Set[str], regex) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = defaultdict(list)
    for file_path in sorted(mod_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if include_ext:
            name_lower = file_path.name.lower()
            if not any(name_lower.endswith(ext) for ext in include_ext):
                continue
        patient_id = _infer_patient_from_filename(file_path, regex)
        mapping[patient_id].append(file_path)
    # Preserve deterministic order
    return {pid: sorted(paths) for pid, paths in mapping.items() if paths}


def _collect_patient_files(
    dataroot: Path,
    modalities: Sequence[str],
    include_ext: Sequence[str],
    id_regex: str,
) -> Dict[str, Dict[str, List[Path]]]:
    regex = _build_id_parser(id_regex)
    include_set = {ext.lower() for ext in include_ext}
    all_mapping: Dict[str, Dict[str, List[Path]]] = {}
    for modality in modalities:
        mod_dir = dataroot / modality
        if not mod_dir.is_dir():
            raise RuntimeError(f"Expected modality directory missing: {mod_dir}")
        mapping = _gather_by_patient_from_subdirs(mod_dir)
        if not mapping:
            mapping = _gather_by_patient_from_files(mod_dir, include_set, regex)
        if not mapping:
            raise RuntimeError(f"No files discovered for modality '{modality}' in {mod_dir}.")
        all_mapping[modality] = mapping
    return all_mapping


def _union_patient_ids(modality_map: Mapping[str, Mapping[str, Sequence[Path]]]) -> List[str]:
    patient_ids: Set[str] = set()
    for mapping in modality_map.values():
        patient_ids.update(mapping.keys())
    patients = sorted(patient_ids)
    if not patients:
        raise RuntimeError("No patient IDs were discovered across the provided modalities.")
    return patients


def _assign_patients_to_folds(patients: Sequence[str], folds: int, seed: int) -> Dict[str, int]:
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    patients = list(patients)
    rng = random.Random(seed)
    rng.shuffle(patients)
    assignments: Dict[str, int] = {}
    for idx, pid in enumerate(patients):
        assignments[pid] = idx % folds
    return assignments


def _ensure_output_root(path: Path, force: bool):
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory '{path}' already exists. Pass --force to remove it before generating folds."
            )
        if not path.is_dir():
            raise FileExistsError(f"Cannot reuse existing non-directory path: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel_src = os.path.relpath(src, dst.parent)
    except Exception:
        rel_src = str(src)
    dst.symlink_to(rel_src)


def _write_manifest(fold_dir: Path, train_ids: Sequence[str], val_ids: Sequence[str]):
    manifest_path = fold_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as fh:
        fh.write("# Patients per split\n")
        fh.write("train:\n")
        for pid in train_ids:
            fh.write(f"  - {pid}\n")
        fh.write("val:\n")
        for pid in val_ids:
            fh.write(f"  - {pid}\n")


# -----------------------------
# Main routine
# -----------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dataroot: Path = args.dataroot.resolve()
    if not dataroot.is_dir():
        parser.error(f"--dataroot must be an existing directory (got {dataroot}).")

    output_root: Path = args.output_root.resolve()

    try:
        modalities = _infer_modalities(dataroot, args.modalities)
    except RuntimeError as exc:
        parser.error(str(exc))

    try:
        modality_map = _collect_patient_files(dataroot, modalities, args.include_ext, args.id_regex)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    patients = _union_patient_ids(modality_map)

    assignments = _assign_patients_to_folds(patients, args.folds, args.seed)

    # Report modality coverage and missing patients
    for modality in modalities:
        missing = sorted(pid for pid in patients if pid not in modality_map[modality])
        if missing:
            print(
                f"[warn] Modality '{modality}' is missing {len(missing)} patient(s): {', '.join(missing)}",
                file=sys.stderr,
            )

    # Summarise plan
    fold_members: Dict[int, List[str]] = defaultdict(list)
    for pid, fold_idx in assignments.items():
        fold_members[fold_idx].append(pid)
    for fold_idx in range(args.folds):
        members = fold_members.get(fold_idx, [])
        print(f"Fold {fold_idx + 1}: {len(members)} patient(s)")
        if len(members) <= 10:
            print("  " + ", ".join(members))

    if args.dry_run:
        print("[dry-run] No directories were created.")
        return 0

    try:
        _ensure_output_root(output_root, args.force)
    except FileExistsError as exc:
        parser.error(str(exc))

    # Materialise directories
    for fold_idx in range(args.folds):
        fold_dir = output_root / f"fold_{fold_idx + 1}"
        val_ids = sorted(fold_members.get(fold_idx, []))
        train_ids = sorted(pid for pid in patients if assignments[pid] != fold_idx)
        for split_name, split_ids in (("train", train_ids), ("val", val_ids)):
            for modality in modalities:
                dest_dir = fold_dir / split_name / modality
                for pid in split_ids:
                    for src in modality_map[modality].get(pid, []):
                        dst = dest_dir / src.name
                        _symlink(src, dst)
        _write_manifest(fold_dir, train_ids, val_ids)

    print(f"Done. Generated {args.folds} fold(s) under '{output_root}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
