# test pretrained weights

```shell
python test.py --config yaml/CycleGAN-pretrained.yaml
```

# train RegGAN

```shell
CUDA_VISIBLE_DEVICES=1 python train.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN).yaml'
CUDA_VISIBLE_DEVICES=1 python train.py --config 'yaml/CycleGAN-noise1-NC+R(RegGAN).yaml'
CUDA_VISIBLE_DEVICES=2 python train.py --config 'yaml/CycleGAN-noise2-NC+R(RegGAN).yaml'
python train.py --config 'yaml/CycleGAN-noise3-NC+R(RegGAN).yaml'
CUDA_VISIBLE_DEVICES=3 python train.py --config 'yaml/CycleGAN-noise4-NC+R(RegGAN).yaml'
CUDA_VISIBLE_DEVICES=2 python train.py --config 'yaml/CycleGAN-noise5-NC+R(RegGAN).yaml'

CUDA_VISIBLE_DEVICES=3 python train.py --config 'yaml/CycleGAN-noise3-NC+R(RegGAN)-trial.yaml'

CUDA_VISIBLE_DEVICES=1 python train.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD.yaml'
CUDA_VISIBLE_DEVICES=2 python train.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch.yaml'
CUDA_VISIBLE_DEVICES=3 python train.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch-keepratio.yaml'

CUDA_VISIBLE_DEVICES=3 python train.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-masked.yaml'

CUDA_VISIBLE_DEVICES=0 python train.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-masked.yaml'
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 train.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-dist.yaml'
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 train.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-masked-dist.yaml'

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 train.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch-keepratio.yaml'

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 train.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold1.yaml'

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 train.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold2.yaml'
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29501 train.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold3.yaml'

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 train.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-foldx.yaml'
```

# test RegGAN

```shell
CUDA_VISIBLE_DEVICES=0 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN).yaml'
CUDA_VISIBLE_DEVICES=0 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD.yaml'
CUDA_VISIBLE_DEVICES=2 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch.yaml'
CUDA_VISIBLE_DEVICES=1 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch-keepratio.yaml'
CUDA_VISIBLE_DEVICES=1 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch-keepratio-bestparams.yaml'
CUDA_VISIBLE_DEVICES=1 python test.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-dist.yaml'
CUDA_VISIBLE_DEVICES=3 python test.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-masked-dist.yaml'

CUDA_VISIBLE_DEVICES=1 python test.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold1.yaml'
CUDA_VISIBLE_DEVICES=2 python test.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold2.yaml'
CUDA_VISIBLE_DEVICES=3 python test.py --config 'yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold3.yaml'
```

# Experiment 01 - Deformation Field

```shell
python experiment01-deform-field.py \
  --config 'yaml/CycleGAN-noise0-NC+R(RegGAN).yaml' \
  --levels 0,1,2,3,4,5 \
  --mode e2e \
  --metric p95 \
  --spacing-mm 0 0 \
  --envelope rss \
  --out_dir 'experiment-results/01-deform-field/CycleGAN_noise0/NC+R/df_eval/'

python experiment01-deform-field.py \
  --config 'yaml/CycleGAN-noise1-NC+R(RegGAN).yaml' \
  --levels 0,1,2,3,4,5 \
  --mode ronly \
  --metric mean \
  --spacing-mm 0 0 \
  --envelope power \
  --out_dir 'experiment-results/01-deform-field/CycleGAN_noise1/NC+R/df_eval/'

python experiment01-deform-field.py \
  --config 'yaml/CycleGAN-noise3-NC+R(RegGAN).yaml' \
  --levels 0,1,2,3,4,5 \
  --mode e2e \
  --metric p95 \
  --spacing-mm 0 0 \
  --envelope rss \
  --out_dir 'experiment-results/01-deform-field/CycleGAN_noise3/NC+R/df_eval/'

python experiment01-deform-field.py \
  --config 'yaml/CycleGAN-noise3-NC+R(RegGAN).yaml' \
  --levels 3 \
  --mode ronly \
  --metric mean \
  --spacing-mm 0 0 \
  --envelope power \
  --viz-all-points \
  --out_dir 'experiment-results/01-deform-field/CycleGAN_noise3/NC+R/df_eval_noise3_T2Dconvert+yxepe/'

python experiment01-deform-field.py \
  --config 'yaml/CycleGAN-noise3-NC+R(RegGAN).yaml' \
  --levels 0,1,2,3,4,5 \
  --mode ronly \
  --metric mean \
  --spacing-mm 0 0 \
  --envelope none \
  --out_dir 'experiment-results/01-deform-field/CycleGAN_noise3/NC+R/df_eval_noise3_T2Dconvert+yxepe/'

python experiment01-deform-field.py \
  --config 'yaml/CycleGAN-noise5-NC+R(RegGAN).yaml' \
  --levels 0,1,2,3,4,5 \
  --mode e2e \
  --metric mean \
  --spacing-mm 0 0 \
  --envelope rss \
  --out_dir 'experiment-results/01-deform-field/CycleGAN_noise5/NC+R/df_eval/'
```

# Experiment 04-2 - Cross Training Test
```shell
CUDA_VISIBLE_DEVICES=3 python experiment04-2-cross-training-test.py \
    --config yaml/RegGAN-SynthRAD-cross3-test.yaml \
    --fold-root output/SynthRAD-RegGAN-512-keepratio-cross3/fold_01 \
    --fold-root output/SynthRAD-RegGAN-512-keepratio-cross3/fold_02 \
    --fold-root output/SynthRAD-RegGAN-512-keepratio-cross3/fold_03 \
    --mode both \
    --compute-fid --fid-feature 64 \
    --compute-lpips --lpips-net vgg \
    --output output/SynthRAD-RegGAN-512-keepratio-cross3/test-eval/cross_eval_summary.json \
    --fig-dir output/SynthRAD-RegGAN-512-keepratio-cross3/test-eval/fig

CUDA_VISIBLE_DEVICES=3 python experiment04-2-cross-training-test.py \
    --config yaml/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold123-test.yaml \
    --fold-root output/SynthRAD-RegGAN-512-keepratio-bestparams-foreground-nomask-fold1 \
    --fold-root output/SynthRAD-RegGAN-512-keepratio-bestparams-foreground-nomask-fold2 \
    --fold-root output/SynthRAD-RegGAN-512-keepratio-bestparams-foreground-nomask-fold3 \
    --mode both \
    --compute-fid --fid-feature 64 \
    --compute-lpips --lpips-net vgg \
    --output output/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold123-test/test-eval/cross_eval_summary.json \
    --fig-dir output/RegGAN-SynthRAD-512-keepratio-bestparams-foreground-nomask-fold123-test/test-eval/fig
```

# Train RegGAN - Fusion mode
```shell
python train.py --config yaml/RegGAN-SynthRAD-fusion3.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-SynthRAD-fusion3.yaml
```

# Test RegGAN - Fusion mode
```shell
CUDA_VISIBLE_DEVICES=0 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-SynthRAD-fusion3.yaml \
      --weights output/SynthRAD-RegGAN-fusion3/NC+R/netG_A2B.pth \
      --reg-weights output/SynthRAD-RegGAN-fusion3/NC+R/R_A.pth \
      --data-root data/SynthRAD2023-Task1/test2D-foreground/ \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-dir output/SynthRAD-RegGAN-fusion3/test-eval/figs \
      --output output/SynthRAD-RegGAN-fusion3/test-eval/summary.json

CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-SynthRAD-fusion3.yaml \
      --weights output/SynthRAD-RegGAN-fusion3/NC+R/best.pth \
      --reg-weights output/SynthRAD-RegGAN-fusion3/NC+R/best.pth \
      --data-root data/SynthRAD2023-Task1/test2D-foreground/ \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-dir output/SynthRAD-RegGAN-fusion3/test-eval/figs \
      --output output/SynthRAD-RegGAN-fusion3/test-eval/summary.json
      
CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-SynthRAD-fusion3.yaml \
      --weights output/SynthRAD-RegGAN-fusion3/NC+R/best.pth \
      --reg-weights output/SynthRAD-RegGAN-fusion3/NC+R/best.pth \
      --data-root data/SynthRAD2023-Task1/test2D-foreground/ \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --output output/SynthRAD-RegGAN-fusion3/test-eval/summary.json
```

# Train RegGAN - Fusion mode with split from body
```shell
python train.py --config yaml/RegGAN-SynthRAD-fusion3-headsplit.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-SynthRAD-fusion3-headsplit.yaml

# Add noise (affine deformation)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-SynthRAD-noise5-fusion3-headsplit.yaml
```

# Test RegGAN - Fusion mode with split from body
```shell
CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-SynthRAD-fusion3-headsplit.yaml \
      --weights output/SynthRAD-RegGAN-fusion3-headsplit/NC+R/best.pth \
      --reg-weights output/SynthRAD-RegGAN-fusion3-headsplit/NC+R/best.pth \
      --data-root data/SynthRAD2023-Task1/test2D-foreground/ \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-dir output/SynthRAD-RegGAN-fusion3-headsplit/test-eval/figs \
      --output output/SynthRAD-RegGAN-fusion3-headsplit/test-eval/summary.json

# Add noise (affine deformation)
CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-SynthRAD-noise5-fusion3-headsplit.yaml \
      --weights output/SynthRAD-noise5-RegGAN-fusion3-headsplit/NC+R/best.pth \
      --reg-weights output/SynthRAD-noise5-RegGAN-fusion3-headsplit/NC+R/best.pth \
      --data-root data/SynthRAD2023-Task1/test2D-foreground/ \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-dir output/SynthRAD-noise5-RegGAN-fusion3-headsplit/test-eval/figs \
      --output output/SynthRAD-noise5-RegGAN-fusion3-headsplit/test-eval/summary.json
```

# Train RegGAN - Fusion mode with split from body CT2MR
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-SynthRAD-CT2MR-fusion3-headsplit.yaml
```

# Test RegGAN - Fusion mode with split from body CT2MR
```shell
CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-SynthRAD-CT2MR-fusion3-headsplit.yaml \
      --weights output/SynthRAD-RegGAN-CT2MR-fusion3-headsplit/NC+R/best.pth \
      --reg-weights output/SynthRAD-RegGAN-CT2MR-fusion3-headsplit/NC+R/best.pth \
      --data-root data/SynthRAD2023-Task1-CT2MR/test2D-foreground/ \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-dir output/SynthRAD-RegGAN-CT2MR-fusion3-headsplit/test-eval/figs \
      --output output/SynthRAD-RegGAN-CT2MR-fusion3-headsplit/test-eval/summary.json
```

# Train RegGAN on SAROPT - Fusion mode with split from body 
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-SAROPT-fusion3-headsplit.yaml
```

# Train RegGAN on SAROPT SAR2OPT - Fusion mode with split from body 
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-SAROPT-SAR2OPT-fusion3-headsplit.yaml
```

# Test RegGAN on SAROPT SAR2OPT - Fusion mode with split from body
```shell
CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-SAROPT-SAR2OPT-fusion3-headsplit.yaml \
      --weights output/SAROPT-SAR2OPT-RegGAN-fusion3-headsplit/NC+R/best.pth \
      --reg-weights output/SAROPT-SAR2OPT-RegGAN-fusion3-headsplit/NC+R/best.pth \
      --data-root data/QXSLAB_SAROPT/test \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-dir output/SAROPT-SAR2OPT-RegGAN-fusion3-headsplit/test-eval/figs \
      --output output/SAROPT-SAR2OPT-RegGAN-fusion3-headsplit/test-eval/summary.json
```

# Train RegGAN on VIPL-MumoFace NIR2RGB - Fusion mode with split from body 
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-NIRVIS-NIR2RGB-fusion3-headsplit.yaml

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 train.py --config yaml/RegGAN-NIRVIS-NIR2RGB-fusion3-headsplit.yaml

python train.py --config yaml/RegGAN-NIRVIS-NIR2RGB-fusion3-headsplit.yaml
```

# Test RegGAN on VIPL-MumoFace NIR2RGB - Fusion mode with split from body
```shell
CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-NIRVIS-NIR2RGB-fusion3-headsplit.yaml \
      --weights output/NIRVIS-NIR2RGB-RegGAN-fusion3-headsplit/NC+R/netG_A2B.pth \
      --reg-weights output/NIRVIS-NIR2RGB-RegGAN-fusion3-headsplit/NC+R/R_A.pth \
      --data-root data/VIPL-MumoFace-2000-merged/test \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-dir output/NIRVIS-NIR2RGB-RegGAN-fusion3-headsplit/test-eval/figs \
      --output output/NIRVIS-NIR2RGB-RegGAN-fusion3-headsplit/test-eval/summary.json
```

# Train RegGAN on OuluCASIA NIR2VIS - Fusion mode with split from body 
```shell
WANDB_DEBUG=true CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-OuluCASIA-Q5-fusion3.yaml

python train.py --config yaml/RegGAN-OuluCASIA-Q5-fusion3.yaml

WANDB_DEBUG=true CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-OuluCASIA-All-fusion3.yaml
```

# Test RegGAN on OuluCASIA NIR2VIS - Fusion mode with split from body
```shell

CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-NIRVIS-NIR2RGB-fusion3-headsplit-testOulu.yaml \
      --weights output/NIRVIS-NIR2RGB-RegGAN-fusion3-headsplit-v251209T2053-resfix/NC+R/netG_A2B.pth \
      --reg-weights output/NIRVIS-NIR2RGB-RegGAN-fusion3-headsplit-v251209T2053-resfix/NC+R/R_A.pth \
      --data-root data/Oulu-CASIA-FE/test/all \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-dir output/NIRVIS-NIR2RGB-RegGAN-fusion3-headsplit-v251209T2053-resfix/test-eval-on-OuluCASIA/figs \
      --output output/NIRVIS-NIR2RGB-RegGAN-fusion3-headsplit-v251209T2053-resfix/test-eval-on-OuluCASIA/summary.json
      
CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-OuluCASIA-Q1-fusion3.yaml \
      --weights output/OuluCASIA-Q1-RegGAN-fusion3/NC+R/netG_A2B.pth \
      --reg-weights output/OuluCASIA-Q1-RegGAN-fusion3/NC+R/R_A.pth \
      --data-root data/Oulu-CASIA-FE/test/all \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-workers 32 \
      --fig-dir output/OuluCASIA-Q1-RegGAN-fusion3/test-eval/figs \
      --output output/OuluCASIA-Q1-RegGAN-fusion3/test-eval/summary.json

CUDA_VISIBLE_DEVICES=2 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-OuluCASIA-Q2-fusion3.yaml \
      --weights output/OuluCASIA-Q2-RegGAN-fusion3/NC+R/netG_A2B.pth \
      --reg-weights output/OuluCASIA-Q2-RegGAN-fusion3/NC+R/R_A.pth \
      --data-root data/Oulu-CASIA-FE/test/all \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-workers 32 \
      --fig-dir output/OuluCASIA-Q2-RegGAN-fusion3/test-eval/figs \
      --output output/OuluCASIA-Q2-RegGAN-fusion3/test-eval/summary.json
      
CUDA_VISIBLE_DEVICES=1 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-OuluCASIA-Q3-fusion3.yaml \
      --weights output/OuluCASIA-Q3-RegGAN-fusion3/NC+R/netG_A2B.pth \
      --reg-weights output/OuluCASIA-Q3-RegGAN-fusion3/NC+R/R_A.pth \
      --data-root data/Oulu-CASIA-FE/test/all \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-workers 32 \
      --fig-dir output/OuluCASIA-Q3-RegGAN-fusion3/test-eval/figs \
      --output output/OuluCASIA-Q3-RegGAN-fusion3/test-eval/summary.json
      
CUDA_VISIBLE_DEVICES=0 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-OuluCASIA-Q4-fusion3.yaml \
      --weights output/OuluCASIA-Q4-RegGAN-fusion3/NC+R/netG_A2B.pth \
      --reg-weights output/OuluCASIA-Q4-RegGAN-fusion3/NC+R/R_A.pth \
      --data-root data/Oulu-CASIA-FE/test/all \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-workers 32 \
      --fig-dir output/OuluCASIA-Q4-RegGAN-fusion3/test-eval/figs \
      --output output/OuluCASIA-Q4-RegGAN-fusion3/test-eval/summary.json
      
CUDA_VISIBLE_DEVICES=3 python experiment05-fusion-training-test.py \
      --config yaml/RegGAN-OuluCASIA-Q5-fusion3.yaml \
      --weights output/OuluCASIA-Q5-RegGAN-fusion3-val1/NC+R/netG_A2B.pth \
      --reg-weights output/OuluCASIA-Q5-RegGAN-fusion3-val1/NC+R/R_A.pth \
      --data-root data/Oulu-CASIA-FE/test/all \
      --compute-fid --fid-feature 64 \
      --compute-lpips --lpips-net vgg \
      --fig-workers 32 \
      --fig-dir output/OuluCASIA-Q5-RegGAN-fusion3-val1/test-eval/figs \
      --output output/OuluCASIA-Q5-RegGAN-fusion3-val1/test-eval/summary.json
```

# Basic RegGAN
```shell
python train.py --config yaml/RegGANBasic-NIRVIS-NIR2RGB.yaml
python test.py --config yaml/RegGANBasic-NIRVIS-NIR2RGB.yaml

CUDA_VISIBLE_DEVICES=1 python test.py --config yaml/RegGANBasic-OuluCASIA-Q1.yaml

python experiment06-basic-training-test.py \
    --config yaml/RegGANBasic-OuluCASIA-Q1.yaml \
    --weights ./output/OuluCASIA-Q1-RegGAN-basic/NC+R/netG_A2B.pth \
    --reg-weights ./output/OuluCASIA-Q1-RegGAN-basic/NC+R/R_A.pth \
    --data-root data/Oulu-CASIA-FE/test/all \
    --compute-fid --fid-feature 64 \
    --compute-lpips --lpips-net vgg \
    --fig-dir ./output/OuluCASIA-Q1-RegGAN-basic/test-eval/figs \
    --output ./output/OuluCASIA-Q1-RegGAN-basic/test-eval/summary.json \
    --fig-workers 4
    
python experiment06-basic-training-test.py \
    --config yaml/RegGANBasic-OuluCASIA-Q2.yaml \
    --weights ./output/OuluCASIA-Q2-RegGAN-basic/NC+R/netG_A2B.pth \
    --reg-weights ./output/OuluCASIA-Q2-RegGAN-basic/NC+R/R_A.pth \
    --data-root data/Oulu-CASIA-FE/test/all \
    --compute-fid --fid-feature 64 \
    --compute-lpips --lpips-net vgg \
    --fig-dir ./output/OuluCASIA-Q2-RegGAN-basic/test-eval/figs \
    --output ./output/OuluCASIA-Q2-RegGAN-basic/test-eval/summary.json \
    --fig-workers 4

python experiment06-basic-training-test.py \
    --config yaml/RegGANBasic-OuluCASIA-Q3.yaml \
    --weights ./output/OuluCASIA-Q3-RegGAN-basic/NC+R/netG_A2B.pth \
    --reg-weights ./output/OuluCASIA-Q3-RegGAN-basic/NC+R/R_A.pth \
    --data-root data/Oulu-CASIA-FE/test/all \
    --compute-fid --fid-feature 64 \
    --compute-lpips --lpips-net vgg \
    --fig-dir ./output/OuluCASIA-Q3-RegGAN-basic/test-eval/figs \
    --output ./output/OuluCASIA-Q3-RegGAN-basic/test-eval/summary.json \
    --fig-workers 4

python experiment06-basic-training-test.py \
    --config yaml/RegGANBasic-OuluCASIA-Q4.yaml \
    --weights ./output/OuluCASIA-Q4-RegGAN-basic/NC+R/netG_A2B.pth \
    --reg-weights ./output/OuluCASIA-Q4-RegGAN-basic/NC+R/R_A.pth \
    --data-root data/Oulu-CASIA-FE/test/all \
    --compute-fid --fid-feature 64 \
    --compute-lpips --lpips-net vgg \
    --fig-dir ./output/OuluCASIA-Q4-RegGAN-basic/test-eval/figs \
    --output ./output/OuluCASIA-Q4-RegGAN-basic/test-eval/summary.json \
    --fig-workers 4

python experiment06-basic-training-test.py \
    --config yaml/RegGANBasic-OuluCASIA-Q5.yaml \
    --weights ./output/OuluCASIA-Q5-RegGAN-basic/NC+R/netG_A2B.pth \
    --reg-weights ./output/OuluCASIA-Q5-RegGAN-basic/NC+R/R_A.pth \
    --data-root data/Oulu-CASIA-FE/test/all \
    --compute-fid --fid-feature 64 \
    --compute-lpips --lpips-net vgg \
    --fig-dir ./output/OuluCASIA-Q5-RegGAN-basic/test-eval/figs \
    --output ./output/OuluCASIA-Q5-RegGAN-basic/test-eval/summary.json \
    --fig-workers 4
```

# Train RegGAN on VIPL-MumoFace NIR2RGB - Fusion mode with split from body, no reg
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config yaml/RegGAN-NIRVIS-NIR2RGB-fusion3-noreg.yaml
```