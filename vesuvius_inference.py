"""
Vesuvius Challenge – Surface Detection Inference Pipeline

Author: Kedar Kale
Model: 2.5D UNet with Spatial Attention
Competition: Kaggle Vesuvius Challenge – Ink Detection

Key Features:
- 2.5D slice context inference
- Residual UNet with spatial attention
- Overlapping tile-based inference
- Robust checkpoint loading (PyTorch 2.6+ safe)
- Morphology-based postprocessing
"""

# ------------------------------------------------
# CELL 1: Imports & Environment
# ------------------------------------------------
import os
import gc
import zipfile
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tifffile
from PIL import Image

from skimage import morphology, measure
from scipy import ndimage

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# ------------------------------------------------
# CELL 2: Configuration Object
# ------------------------------------------------
class Settings:
    DATA_ROOT = Path('/kaggle/input/vesuvius-challenge-surface-detection')
    TEST_TABLE = DATA_ROOT / 'test.csv'
    TEST_VOLUMES = DATA_ROOT / 'test_images'

    CHECKPOINT = '/kaggle/input/unet2-5segment/pytorch/v2/1/checkpoint_epoch_20.pth'

    SLICE_CONTEXT = 7           # 2.5D depth
    CLASSES = 3
    BASE_WIDTH = 24
    TILE = 128

    BATCH = 64
    WORKERS = 2
    OVERLAP_RATIO = 0.25

    OUT_DIR = Path('/kaggle/working/submission_tifs')
    ZIP_NAME = 'submission.zip'

CFG = Settings()
CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
print("✓ Settings ready")

# ------------------------------------------------
# CELL 3: Volume Loading & Normalization
# ------------------------------------------------
def read_multiframe_tiff(path: Path) -> np.ndarray:
    frames = []
    with Image.open(path) as img:
        try:
            while True:
                frames.append(np.array(img.copy()))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    return np.stack(frames, axis=0)


def robust_normalize(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo + 1e-8)).astype(np.float32)

print("✓ I/O utilities ready")

# ------------------------------------------------
# CELL 4: Network Building Blocks (CORRECTED)
# ------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.query = nn.Conv2d(ch, ch // 8, 1)
        self.key = nn.Conv2d(ch, ch // 8, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).reshape(b, -1, h * w).transpose(1, 2)
        k = self.key(x).reshape(b, -1, h * w)
        v = self.value(x).reshape(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.gamma * out


class ResidualConv(nn.Module):
    def __init__(self, cin, cout, attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        
        # FIX: Renamed 'self.attn' to 'self.attention' to match checkpoint keys
        self.attention = SpatialAttention(cout) if attention else None

    def forward(self, x):
        r = self.residual(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # FIX: Use the new name here as well
        if self.attention:
            x = self.attention(x)
            
        return self.act(x + r)

# (The rest of Cell 4, specifically UNet25D, remains the same)
class UNet25D(nn.Module):
    def __init__(self, in_ch, num_classes, base):
        super().__init__()
        Block = ResidualConv 

        self.enc1 = Block(in_ch, base)
        self.enc2 = Block(base, base * 2)
        self.enc3 = Block(base * 2, base * 4)
        self.enc4 = Block(base * 4, base * 8, attention=True)
        self.bottleneck = Block(base * 8, base * 16, attention=True) 

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = Block(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = Block(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = Block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = Block(base * 2, base)

        self.out = nn.Conv2d(base, num_classes, 1)
        self.out_deep3 = nn.Conv2d(base * 4, num_classes, 1)
        self.out_deep2 = nn.Conv2d(base * 2, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c  = self.bottleneck(self.pool(e4)) 

        d4 = self.dec4(torch.cat([self.up4(c), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.out(d1)

print("✓ Network Blocks ready")

# ------------------------------------------------
# CELL 5: Robust Weights Loading (Fixed for PyTorch 2.6+)
# ------------------------------------------------
def load_checkpoint_safe(model, path):
    # FIX: weights_only=False handles the UnpicklingError
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # 1. Remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        # Fix for num_batches_tracked: Only in BatchNorm layers, which the model uses
        if 'num_batches_tracked' not in k: 
             new_state_dict[name] = v

    try:
        # Load with strict=True now that layer names (bottleneck) and blocks (ResidualConv) are matched
        model.load_state_dict(new_state_dict, strict=True)
        print("✓ Checkpoint loaded successfully (Strict Mode)")
    except RuntimeError as e:
        print("\n! Final check failed. If there are only a few 'num_batches_tracked' warnings, proceed. Otherwise, a key mismatch remains.")
        # If the failure is only due to num_batches_tracked (which is often fine)
        if all('num_batches_tracked' in msg for msg in str(e).split('\n')):
            print("! Warning: The remaining errors are likely minor BatchNorm tracking errors.")
            model.load_state_dict(new_state_dict, strict=False)
        else:
            raise e

net = UNet25D(CFG.SLICE_CONTEXT, CFG.CLASSES, CFG.BASE_WIDTH).to(DEVICE)
load_checkpoint_safe(net, CFG.CHECKPOINT)
net.eval()
print("✓ Checkpoint loaded")

# ------------------------------------------------
# CELL 6: Inference Dataset
# ------------------------------------------------
class VolumeTiles(Dataset):
    def __init__(self, vid: str, path: Path):
        self.id = vid
        self.vol = robust_normalize(read_multiframe_tiff(path))
        self.D, self.H, self.W = self.vol.shape
        self.tiles = self._build_tiles()

    def _build_tiles(self):
        step = int(CFG.TILE * (1 - CFG.OVERLAP_RATIO))
        half = CFG.SLICE_CONTEXT // 2
        coords = []
        for z in range(half, self.D - half):
            for y in range(0, self.H, step):
                for x in range(0, self.W, step):
                    coords.append((z, y, x))
        return coords

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        z, y, x = self.tiles[idx]
        h = min(CFG.TILE, self.H - y)
        w = min(CFG.TILE, self.W - x)
        patch = self.vol[z - 3:z + 4, y:y + h, x:x + w]
        pad_h, pad_w = CFG.TILE - h, CFG.TILE - w
        if pad_h or pad_w:
            patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        return torch.from_numpy(patch).float(), (z, y, x, h, w)

print("✓ Dataset ready")

# ------------------------------------------------
# CELL 7: Inference & Aggregation
# ------------------------------------------------
@torch.no_grad()
def infer_volume(vid: str, path: Path) -> np.ndarray:
    ds = VolumeTiles(vid, path)
    dl = DataLoader(ds, CFG.BATCH, False, num_workers=CFG.WORKERS)
    acc = np.zeros((ds.D, ds.H, ds.W, CFG.CLASSES), np.float32)
    cnt = np.zeros((ds.D, ds.H, ds.W), np.float32)

    for x, meta in tqdm(dl, leave=False):
        x = x.to(DEVICE)
        p = F.softmax(net(x), 1).cpu().numpy()
        for i, (z, y, x0, h, w) in enumerate(zip(*meta)):
            z, y, x0, h, w = int(z), int(y), int(x0), int(h), int(w)
            acc[z, y:y + h, x0:x0 + w] += p[i, :, :h, :w].transpose(1, 2, 0)
            cnt[z, y:y + h, x0:x0 + w] += 1

    return np.argmax(acc / np.maximum(cnt[..., None], 1), -1).astype(np.uint8)

print("✓ Inference routine ready")

# ------------------------------------------------
# CELL 8: Scorer‑Safe Postprocessing
# ------------------------------------------------
def simplify(mask: np.ndarray) -> np.ndarray:
    mask = morphology.remove_small_objects(mask.astype(bool), 3000, connectivity=3)
    lbl = measure.label(mask, connectivity=3)
    if lbl.max() > 10:
        sizes = ndimage.sum(mask, lbl, range(1, lbl.max() + 1))
        keep = np.argsort(sizes)[-10:] + 1
        mask = np.isin(lbl, keep)
    kernel = morphology.ball(2)
    mask = morphology.binary_closing(mask, kernel)
    mask = morphology.binary_opening(mask, kernel)
    mask = morphology.remove_small_holes(mask, 5000)
    return mask.astype(np.uint8)

print("✓ Postprocessing ready")

# ------------------------------------------------
# CELL 9: Main Loop
# ------------------------------------------------
meta = pd.read_csv(CFG.TEST_TABLE)
for row in meta.itertuples():
    path = CFG.TEST_VOLUMES / f"{row.id}.tif"
    pred = infer_volume(row.id, path)
    pred = simplify(pred)
    tifffile.imwrite(CFG.OUT_DIR / f"{row.id}.tif", pred)
    gc.collect()

print("✓ All volumes processed")

# ------------------------------------------------
# CELL 10: Zip Submission
# ------------------------------------------------
with zipfile.ZipFile(CFG.ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in CFG.OUT_DIR.glob('*.tif'):
        z.write(f, f.name)

print("✓ Submission archive created")