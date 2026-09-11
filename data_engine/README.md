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
| 3 | `step3_add_poisson_noise.py` | Adds randomised signal-dependent Poisson noise. |

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

Each tree is rendered at several timesteps (`t_start`..`t_end` in steps of
`t_step`), so one tree yields several projections at different contrast phases.

### Step 2 — background fusion

Two randomly drawn real contrast-free frames are combined by MixUp and jittered
in contrast and gamma to diversify background texture. The vessel projection is
randomly scaled, shifted and rotated, then fused as `I_t = B - alpha * V_t`,
where `alpha` controls vessel visibility.

Every sample writes **two** contrast-filled images over the **same** background
with **different** vessel trees. That background invariance is what the
vessel-aware contrastive loss exploits during training.

### Step 3 — acquisition noise

Poisson noise is applied in three regimes: none (20%), a realisation shared
between the paired images and the background (40%), and an independent
realisation per image (40%). Sharing matters: if only the contrast-filled images
were noisy, the network could detect vessels from the noise pattern instead of
the anatomy.

## Usage

```bash
pip install -r ../requirements.txt -r requirements.txt

python step1_generate_projections.py \
    --config config/settings.yaml \
    --reference-volume /path/to/reference_ct.nii.gz \
    --output-dir output --num-samples 2000

python step2_compose_xca.py \
    --vessels output/projections \
    --backgrounds /path/to/contrast_free_frames \
    --output ../data/synthetic/train_clean --num-samples 50000

python step3_add_poisson_noise.py \
    --input ../data/synthetic/train_clean --output ../data/synthetic/train
```

### Inputs you need to supply

* **A reference NIfTI volume.** Only its header, affine and dimensions are used,
  to define the synthesis grid and the DRR geometry. Any CT volume of the region
  of interest works.
* **A pool of real contrast-free XCA frames.** Frames captured before contrast
  injection, cropped to remove burned-in annotations. These provide realistic
  background anatomy and detector texture; without them the synthetic images look
  too clean and the domain gap widens.

## Configuration

All sampling ranges live in `config/settings.yaml`, documented inline:
tree morphology (`vessel_generation`), contrast dynamics
(`contrast_dynamics`) and projection geometry (`projection`). Widening the
ranges increases diversity at the cost of realism; the shipped values are the
ones used for the paper.

## Cost

Step 1 is the bottleneck: it is dominated by voxel traversal and DRR rendering,
and it needs a GPU for `diffdrr`. Steps 2 and 3 are CPU-only and cheap. The paper
trains on 50,000 composed pairs; the ablation shows performance rising
monotonically from 5k to 50k, so more data is worth the compute.

## License

`vsystem/` is a trimmed and extended copy of
[V-System](https://github.com/psweens/V-System), licensed under **GPL-3.0**
(`vsystem/LICENSE.txt`). See `../THIRD_PARTY_NOTICES.md`.
