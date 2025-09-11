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
```

# test RegGAN

```shell
CUDA_VISIBLE_DEVICES=0 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN).yaml'
CUDA_VISIBLE_DEVICES=0 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD.yaml'
CUDA_VISIBLE_DEVICES=2 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch.yaml'
CUDA_VISIBLE_DEVICES=1 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch-keepratio.yaml'
CUDA_VISIBLE_DEVICES=1 python test.py --config 'yaml/CycleGAN-noise0-NC+R(RegGAN)-SynthRAD-bigbatch-keepratio-bestparams.yaml'
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