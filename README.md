# Cellular electrophysiology signal reconstruction

Reconstructor of cellular electrophysiology signal segments with a 1D U-Net-style PyTorch model. Maps extracellular signal windows to paired intracellular signal windows, and predicts action potential duration-width targets from extracellular windows.

## Method

- 1D convolution + batch normalization + ReLU blocks.
- Residual blocks with squeeze/excitation-style channel recalibration.
- Encoder stages with stride-based downsampling.
- Nearest-neighbor upsampling with skip concatenation.
- Optional dense projection from trace length to 15 APD widths.

## Train

```bash
python scripts/train.py --task extra_to_intra --epochs 30 --batch-size 32 --lr 0.0001
```

- `--task intra_to_extra`: reverses the paired root arrays.
- `--task apd`: uses dense APD projection.