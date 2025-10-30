from pathlib import Path

root = Path("data/SynthRAD2023-Task1/cv_folds")
for fold in sorted(root.glob("fold_*")):
    path = fold / "val" / "B" / "1PC042.nii_z0000.npy"
    print(f"{fold.name}: {'found' if path.is_file() else 'missing'}")