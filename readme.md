# TDBM: Temporal Diffusion Bridge Model for Vessel Extraction Using Physics-informed Synthetic Data

Official implementation of *Temporal Diffusion Bridge Model for Vessel Extraction
Using Physics-informed Synthetic Data* (Under review).

Jinkui Hao, Donald R. Cantrell, Ramez Abdalla, Sameer A. Ansari, Bo Zhou
· Department of Radiology, Northwestern University

---

## Overview

Reliable vessel extraction from X-ray coronary angiography (XCA) is hard: vessel
visibility is low, and respiratory and cardiac motion break the assumptions of
conventional digital subtraction angiography. Supervised deep models work, but
need large hand-annotated datasets, and most of them return only a binary mask,
discarding the intra-arterial contrast information clinicians actually read.

TDBM takes a different route. Vessel extraction is posed as **image-to-image
translation**: predict the *contrast-free* frame from a contrast-filled sequence.
The vessels are then recovered as the residual, with their contrast intensity
intact. Training needs **no human annotations** — the supervision comes from a
physics-based XCA synthesis engine that produces perfectly aligned pairs of
contrast-filled and contrast-free images.

The method has three parts:

1. **Physics-based data engine** (`data_engine/`). A stochastic Lindenmayer
   system grows 3-D arterial trees whose bifurcation angles and diameter tapering
   obey Murray's law; contrast propagation is simulated with per-segment flow
   velocities and a gamma-variate enhancement profile; the resulting 4-D field is
   voxelised and rendered with a differentiable DRR; and the projections are
   fused with real contrast-free fluoroscopic backgrounds.
2. **Temporal Diffusion Bridge Model** (`models/`). A Brownian bridge between the
   contrast-filled and contrast-free domains, driven by an attention U-Net that
   is conditioned on a temporal window of neighbouring frames for motion
   consistency.
3. **Vessel-aware contrastive loss** (`losses.py`). The engine can render two
   images that share a background but differ in vasculature; contrasting their
   intermediate features against background features sharpens the model's
   response to vessels and suppresses residual vessel signal in the output.

---

## Installation

```bash
git clone https://github.com/Advanced-AI-in-Medicine-and-Physics-Lab/TDBM.git
cd TDBM

conda create -n tdbm python=3.9 -y
conda activate tdbm

pip install -r requirements.txt
# Only needed if you want to generate synthetic data yourself:
pip install -r data_engine/requirements.txt
```

Tested with Python 3.9, PyTorch 2.7 and CUDA 12.6 on NVIDIA A100 GPUs.

---

## Quick start: extract vessels from your own sequences

Arrange your frames as one directory per sequence, then run:

```bash
python inference.py \
    --config config/tdbm.json \
    --checkpoint /path/to/tdbm.pth \
    --input  /path/to/sequences \
    --output results/my_study
```

```
/path/to/sequences/
├── case_001/
│   ├── frame_000.png
│   ├── frame_001.png
│   └── ...
└── case_002/
    └── ...
```

`results/my_study/residual/` mirrors that structure and holds the extracted
vessels (dark on a bright background), preserving the intra-arterial contrast
intensity. See [Inference](#inference) for the available options.

---

## Data

`docs/DATA.md` documents every directory layout the code expects. In short:

| Purpose | Config key | Produced by |
| --- | --- | --- |
| Synthetic training pairs | `synthetic_data_root` | `data_engine/` (3 steps below) |
| Real XCA sequences (annealed term) | `real_data_root` | your own data |
| Validation set with annotations | `val_image_root`, `val_gt_root` | e.g. the public XCA30 benchmark |

### Generating synthetic training data

The engine runs in three steps. Steps 1 and 2 need one reference CT volume (for
the synthesis grid geometry) and a pool of real XCA frames captured *before*
contrast injection, which serve as backgrounds.

```bash
cd data_engine

# 1. Vessel trees -> contrast dynamics -> volumes -> DRR projections
python step1_generate_projections.py \
    --config config/settings.yaml \
    --reference-volume /path/to/reference_ct.nii.gz \
    --output-dir output --num-samples 2000

# 2. Fuse projections with MixUp-ed real backgrounds into paired training images
python step2_compose_xca.py \
    --vessels output/projections \
    --backgrounds /path/to/contrast_free_frames \
    --output ../data/synthetic/train_clean --num-samples 50000

# 3. Add randomised Poisson acquisition noise
python step3_add_poisson_noise.py \
    --input ../data/synthetic/train_clean \
    --output ../data/synthetic/train
```

Sampling ranges for tree morphology, contrast dynamics and projection geometry
live in `data_engine/config/settings.yaml`; every parameter is documented inline.
Each vessel tree is rendered at several timesteps, so step 1 produces more
projections than trees.

---

## Training

```bash
python train.py --config config/tdbm.json
```

Point `synthetic_data_root`, `real_data_root`, `val_image_root` and
`val_gt_root` in the config at your data first. Checkpoints (EMA weights) are
written to `outputs/<experiment_name>/`; `model_latest.pth` every epoch and
`model_best.pth` whenever validation Dice improves.

To resume:

```bash
python train.py --config config/tdbm.json --resume outputs/TDBM/model_latest.pth
```

A SLURM template is provided in `scripts/train.slurm`.

### Key configuration options

| Key | Default | Meaning |
| --- | --- | --- |
| `neighbor_num` | `3` | Frames in the temporal window (`2*delta + 1`). The paper's ablation finds 3 optimal; 5 and 7 degrade slightly. |
| `imgSize` | `256` | Training resolution. |
| `use_vcl` / `vcl_weight` | `true` / `0.01` | Vessel-aware contrastive loss and its weight `alpha`. |
| `use_real_data` | `true` | Enable the annealed real-data supervision term. |
| `real_weight_start/end` | `0.5` / `0.25` | Bounds of the annealing schedule `lambda(t)`. |
| `sample_step` | `20` | Reverse diffusion steps `K` used at inference. |
| `num_epochs` | `30` | Training length; the LR follows a cosine schedule. |

Changing `neighbor_num` changes the U-Net input width (`in_channel = neighbor_num + 1`),
so a checkpoint can only be loaded with the `neighbor_num` it was trained with.

---

## Inference

```bash
python inference.py --config config/tdbm.json --checkpoint model_best.pth \
    --input data/XCA30/Images --output results/xca30 \
    --mode multicrop --num-crops 3 --crop-size 224 --resize-to 448
```

Pretrained weights are released — see the
[Releases page]([https://github.com/Advanced-AI-in-Medicine-and-Physics-Lab/TDBM/releases](https://drive.google.com/file/d/1QIe9pZbMrVy2MGCho-rEF7ijL1jXttU-/view?usp=drive_link)).
Download the checkpoint and pass it via `--checkpoint`, together with the config
whose `neighbor_num` matches the one it was trained with.

| Flag | Effect |
| --- | --- |
| `--mode multicrop` | Runs an `n x n` grid of overlapping crops and fuses the residuals with a pixel-wise minimum. Preserves thin distal vessels; used for all reported numbers. |
| `--mode full` | One pass per frame at the model's native resolution. Faster; also writes the reconstructed contrast-free frames. |
| `--sample-step K` | Overrides the number of reverse steps (default `20`, from the config). The paper's acceleration study reports `K = 50` matching the full 1000-step schedule (Dice 0.755), `K = 10` still at 0.753, and a collapse at `K = 2`. |

Outputs (`--mode full` writes all three, `--mode multicrop` writes `residual/`):

```
<output>/reconstruction/   predicted contrast-free frames
<output>/vessel/           extracted vessels
<output>/residual/         extracted vessels, 2x contrast (input to evaluate.py)
```

---

## Evaluation

```bash
python evaluate.py \
    --gt   data/XCA30/GT \
    --pred results/xca30/residual \
    --frames 2 \
    --save-binary results/xca30/binary
```

Reports Dice, sensitivity, specificity and accuracy, and writes
`average_metrics.txt` plus a per-sequence CSV. `--frames` selects which frames of
each sequence to score; the public XCA30 benchmark annotates the third frame,
hence index `2`.



---

## Repository layout

```
TDBM/
├── train.py                  training entry point
├── inference.py              vessel extraction (whole-frame and grid-crop)
├── evaluate.py               segmentation metrics
├── losses.py                 vessel-aware contrastive loss
├── dataset_xca.py            synthetic-pair, real-sequence and inference loaders
├── utils.py                  EMA, annealing schedule, model/config helpers
├── models/
│   ├── bridge.py             Brownian bridge forward/reverse process
│   ├── unet.py               conditional attention U-Net with feature taps
│   ├── nn.py, utils.py       building blocks
├── config/tdbm.json          main configuration
├── data_engine/              physics-based XCA synthesis engine
│   ├── step1_generate_projections.py
│   ├── step2_compose_xca.py
│   ├── step3_add_poisson_noise.py
│   ├── config/settings.yaml
│   ├── vsystem/              stochastic L-system vessel generator (GPL-3.0)
│   └── util/                 contrast dynamics and I/O helpers
├── scripts/                  SLURM job templates
└── docs/DATA.md              dataset layouts
```

---

## Citation

```bibtex
@article{hao2026tdbm,
  title   = {Temporal Diffusion Bridge Model for Vessel Extraction Using
             Physics-informed Synthetic Data},
  author  = {Hao, Jinkui and Cantrell, Donald R. and Abdalla, Ramez and
             Ansari, Sameer A. and Zhou, Bo},
  journal = {XXX},
  year    = {2026}
}
```

## License

Distributed under the **GNU General Public License v3.0** (see `LICENSE`). The
data engine builds on the GPL-3.0-licensed
[V-System](https://github.com/psweens/V-System); `THIRD_PARTY_NOTICES.md` lists
every bundled or derived third-party component and its license.

## Acknowledgements

This work builds on [V-System](https://github.com/psweens/V-System),
[BBDM](https://github.com/xuekt98/BBDM),
[guided-diffusion](https://github.com/openai/guided-diffusion) /
[Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models),
[CUT](https://github.com/taesungp/contrastive-unpaired-translation) and
[DiffDRR](https://github.com/eigenvivek/DiffDRR).
