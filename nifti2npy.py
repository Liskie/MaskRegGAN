import os, glob, numpy as np, nibabel as nib
from pathlib import Path
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

import fire

def dump_slices(vol_glob="/home/yjwang/datasets/SynthRAD2023-Task1-compact/pelvis/train-minmax_sym-norm/mr",
                out_dir="data/SynthRAD2023-Task1/train2D/A",
                reorient='canonical',
                rotate_k=1,
                flip_lr=False,
                flip_ud=False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    vol_glob_in = vol_glob
    if os.path.isdir(vol_glob_in):
        # If a directory is passed, look for NIfTI files inside (non-recursive, then recursive as fallback)
        vol_paths = sorted(glob.glob(os.path.join(vol_glob_in, "*.nii*")))
        if not vol_paths:
            vol_paths = sorted(glob.glob(os.path.join(vol_glob_in, "**", "*.nii*"), recursive=True))
    else:
        # Support glob patterns like **/*.nii.gz
        vol_paths = sorted(glob.glob(vol_glob_in, recursive=True))

    print(f"Found {len(vol_paths)} volumes from '{vol_glob_in}'")
    print(f"Reorient={reorient}, rotate_k={rotate_k}, flip_lr={flip_lr}, flip_ud={flip_ud}")
    if not vol_paths:
        print("No NIfTI volumes found. Tip: pass a directory or a glob like '/path/**/*.nii.gz'.")
        return

    # Pre-compute total number of slices for progress
    total_slices = 0
    for vp in vol_paths:
        try:
            img = nib.load(vp)
            if reorient == 'canonical':
                img = nib.as_closest_canonical(img)
            shp = img.shape  # does not load data
            if len(shp) < 3:
                continue
            total_slices += shp[2]
        except Exception:
            # skip problematic files
            continue

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Converting slices", total=total_slices)

        for vp in vol_paths:
            try:
                img = nib.load(vp)
                if reorient == 'canonical':
                    img = nib.as_closest_canonical(img)
                vol = img.get_fdata(dtype=np.float32)  # (X, Y, Z[, T])
                if vol.ndim >= 4:
                    vol = vol[..., 0]
            except Exception as e:
                # Skip unreadable files
                continue

            z = vol.shape[2] if vol.ndim >= 3 else 0
            stem = Path(vp).stem

            for k in range(z):  # axial slices
                sl = vol[:, :, k]
                # Apply orientation adjustments for display
                if rotate_k in (1, 2, 3):
                    sl = np.rot90(sl, k=rotate_k)
                if flip_lr:
                    sl = np.fliplr(sl)
                if flip_ud:
                    sl = np.flipud(sl)
                np.save(f"{out_dir}/{stem}_z{k:04d}.npy", sl)
                progress.advance(task)
        done = progress.tasks[0].completed if progress.tasks else 0
        print(f"Saved {int(done)} slices to '{out_dir}'.")

if __name__ == '__main__':
    fire.Fire(dump_slices)

    """
    Example usages:
    1) Directory input (auto-search .nii*), canonical reorient, rotate 90° to make slices "upright":
       python nifti2npy.py dump_slices \
         --vol_glob="/home/yjwang/datasets/SynthRAD2023-Task1-compact/pelvis/train-minmax_sym-norm/ct" \
         --out_dir="data/SynthRAD2023-Task1/train2D/B" \
         --reorient=canonical --rotate_k=1
       python nifti2npy.py dump_slices \
         --vol_glob="/home/yjwang/datasets/SynthRAD2023-Task1-compact/pelvis/train-minmax_sym-norm/mr" \
         --out_dir="data/SynthRAD2023-Task1/train2D/A" \
         --reorient=canonical --rotate_k=1
       python nifti2npy.py dump_slices \
         --vol_glob="/home/yjwang/datasets/SynthRAD2023-Task1-compact/pelvis/test-minmax_sym-norm/ct" \
         --out_dir="data/SynthRAD2023-Task1/test2D/B" \
         --reorient=canonical --rotate_k=1
       python nifti2npy.py dump_slices \
         --vol_glob="/home/yjwang/datasets/SynthRAD2023-Task1-compact/pelvis/test-minmax_sym-norm/mr" \
         --out_dir="data/SynthRAD2023-Task1/test2D/A" \
         --reorient=canonical --rotate_k=1

    2) If your viewer needs no rotation:
       python nifti2npy.py dump_slices --vol_glob=".../mr" --out_dir=".../A" --rotate_k=0

    3) Fine-tune flips if needed:
       python nifti2npy.py dump_slices --vol_glob=".../mr" --out_dir=".../A" --rotate_k=1 --flip_lr=True
    """
