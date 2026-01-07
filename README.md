# Vesuvius Challenge – 2.5D UNet Inference (link: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)

This repository contains a **production-grade inference pipeline**
for the Kaggle **Vesuvius Challenge – Surface Detection** competition.

## Highlights
- 2.5D UNet architecture with spatial attention
- Overlapping tile-based inference for large volumes
- Robust checkpoint loading compatible with PyTorch 2.6+
- Morphology-based postprocessing for scorer-safe masks
- Automatic TIFF submission packaging

## How to Run (Kaggle)
1. Upload this repo as a Kaggle Dataset
2. Set checkpoint path in `Settings`
3. Run:

```bash
python vesuvius_inference.py
```

Output:
```
submission.zip
```
