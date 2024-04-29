## ProDiffAD

This repository is the official implementation of ProDiffAD: Progressively Distilled Diffusion Models for Multivariate Time Series Anomaly Detection in JointCloud Environment

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the models in the paper, run this command:

```pyth
python train_diffusion_val.py --training diffusion
```

## Distillation

To distill model from teacher to student:
```
python train_diffusion_val.py --training distill
```

## Inference

To perform inference (by onnx):

```sql
python train_diffusion_val.py --input <model> --use_onnx True (--onnx_name  <path_to_onnxfile>)
```

For example:

```python
python train --dataset point_global --denoise_steps 64 --batch_size 8 --training distill --lr 1e-4 --epoch 5 --train_loss_begin 0.10 --window_size 64 --noise_steps 512 --input point_global_128_128_trial --output point_global_128_64_trial --use_onnx True --onnx_name 1.onnx --test_only True 
```
