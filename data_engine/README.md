# Physics-based XCA Data Engine

Generates the annotation-free training corpus for TDBM: large-scale, perfectly
aligned pairs of contrast-filled and contrast-free XCA-like images.

Unlike generative models that learn appearance priors from data, this engine
reproduces the angiographic formation process mechanistically, which is what
gives it anatomical control and lets the resulting model transfer to unseen
modalities without fine-tuning.

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `step1_generate_projections.py` | Grows a 3-D arterial tree with a stochastic L-system under Murray's-law constraints, simulates contrast propagation, voxelises the 4-D field and renders it with a differentiable DRR. |
| 2 | `step2_compose_xca.py` | Fuses vessel projections with MixUp-ed real contrast-free backgrounds into paired training images. |
| 3 | `step3_add_poisson_noise.py` | Adds randomised, zero-mean, signal-dependent acquisition noise (darker = noisier, calibrated on real frames). |

### Step 1 — vessel trees and projections

1. **Morphology.** A stochastic Lindenmayer system encodes the hierarchical
   branching structure. Parameters are sampled within physiologically plausible
   ranges, and bifurcation angles and diameter tapering follow relationships
   derived from Murray's law, so mass conservation and hydrodynamic resistance
   stay coherent across vessel levels.
2. **Contrast dynamics.** Each segment gets a flow velocity consistent with its
   diameter. The arrival time at position `r` is the cumulative path integral
   `t0(r) = sum_s L_s / v_s` from the injection site, and the local intensity
   follows a gamma-variate profile
   `I(r, t) = A (t - t0)^alpha exp(-(t - t0) / beta)`, which reproduces the
   asymmetric enhancement curve including the physiological washout tail.
3. **Voxelisation.** Segments are rasterised by voxel traversal according to
   their diameter and instantaneous contrast concentration.
4. **Projection.** A DRR integrates attenuation along rays onto a virtual
   detector. Randomised C-arm rotations and translations emulate varying
   fluoroscopic viewpoints and patient positioning.

Each tree is rendered at several **contrast phases** chosen relative to its own
dynamics (`contrast_dynamics.t_mode: relative`): `frames_per_tree` phases
`p ~ U(phase_range)` are rendered at `t = p * (max_arrival_time + alpha * beta)`,
so every tree contributes an early-filling frame (proximal segments only), a
peak frame and washout frames in which the trunk fades before the distal
branches. Two further controls diversify the spatial distribution of contrast:
`branch_velocity_jitter` (log-normal per-branch velocity factor: sibling branches
opacify at different times, filling is never perfectly symmetric) and
`dispersion_gain` (the gamma-variate time constant grows with the arrival time,
so distal curves are broader, lower and later). All phases of one tree are
normalised with a shared intensity scale (`projection.normalise: per_tree`), so
the brightness of a frame reflects its phase; `per_frame` restores the original
per-frame stretching, and `t_mode: absolute` the original fixed time grid.

### Step 2 — background fusion

Two randomly drawn real contrast-free frames are combined by MixUp and jittered
in contrast and gamma to diversify background texture. The vessel projection is
randomly scaled, shifted and rotated, then fused as `I_t = B - alpha * V_t`,
where `alpha` controls vessel visibility.

Every sample writes **two** contrast-filled images over the **same** background
with **different** vessel trees. That background invariance is what the
vessel-aware contrastive loss exploits during training.

### Step 3 — acquisition noise

Zero-mean Gaussian noise whose standard deviation follows the log-processed X-ray
law `sigma(I) = sigma0 * (128 / clip(I, 96, 255))^0.85` (flat in the dark range,
decreasing towards bright pixels; `sigma0` is drawn per image from 1.2-3.6 gray
levels; fitted on 150 real contrast-free frames). Three regimes: none (20%), a realisation shared between
the paired images and the background (40%), and an independent realisation per
image (40%). Sharing matters: if only the contrast-filled images were noisy, the
network could detect vessels from the noise pattern instead of the anatomy. In
the shared regime the field is drawn from the vessel-free background, so it is
identical on all four images and carries no vessel information.

`--model poisson-legacy` reproduces the earlier implementation for comparison. It
added a raw Poisson *sample* to the image, which was mostly a brightness bias
(8-24 %), had variance growing with brightness, and in the shared regime
imprinted a vessel-shaped dip of 8-24 % of the vessel attenuation into the
background targets.

## Usage

```bash
pip install -r ../requirements.txt -r requirements.txt

python step1_generate_projections.py \
    --config config/settings.yaml \
    --output-dir output --num-samples 2000

python step2_compose_xca.py \
    --vessels output/projections \
    --backgrounds /path/to/contrast_free_frames \
    --output ../data/synthetic/train_clean --num-samples 50000

python step3_add_poisson_noise.py \
    --input ../data/synthetic/train_clean --output ../data/synthetic/train
```

### Inputs you need to supply

* **A pool of real contrast-free XCA frames.** Frames captured before contrast
  injection, cropped to remove burned-in annotations. These provide realistic
  background anatomy and detector texture; without them the synthetic images look
  too clean and the domain gap widens.

No reference CT is required: the synthesis grid (voxel count and spacing) is
defined in `config/settings.yaml` (`grid` section, 256^3 voxels of 1 mm by
default). `--reference-volume` can still be passed to copy the grid geometry
from an existing NIfTI volume.

## Configuration

All sampling ranges live in `config/settings.yaml`, documented inline: the
synthesis grid (`grid`), tree morphology (`vessel_generation`), contrast dynamics
(`contrast_dynamics`) and projection geometry (`projection`). Widening the
ranges increases diversity at the cost of realism; the shipped values are the
ones used for the paper.

## Cost

Step 1 is the bottleneck: it is dominated by voxel traversal and DRR rendering,
and it needs a GPU for `diffdrr`. Timesteps at which the volume holds no contrast
(before arrival or after the washout tail) are skipped rather than rendered as
empty frames. Steps 2 and 3 are CPU-only and cheap. The paper
trains on 50,000 composed pairs; the ablation shows performance rising
monotonically from 5k to 50k, so more data is worth the compute.

## License

`vsystem/` is a trimmed and extended copy of
[V-System](https://github.com/psweens/V-System), licensed under **GPL-3.0**
(`vsystem/LICENSE.txt`). See `../THIRD_PARTY_NOTICES.md`.
