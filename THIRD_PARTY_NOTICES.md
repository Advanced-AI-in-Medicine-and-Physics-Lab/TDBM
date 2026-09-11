# Third-Party Notices

This repository bundles or derives from the following third-party work.

## V-System (`data_engine/vsystem/`)

The stochastic Lindenmayer-system vessel generator is a trimmed copy of
[V-System](https://github.com/psweens/V-System) by Paul W. Sweeney et al., which
implements the stochastic L-system grammars of
[Galarreta-Valverde et al. (2013)](https://doi.org/10.1117/12.2007532).

* Upstream license: **GNU General Public License v3.0** — retained verbatim in
  `data_engine/vsystem/LICENSE.txt`.
* Upstream reference: E. L. Brown, T. L. Lefebvre, P. W. Sweeney et al.,
  *Quantification of vascular networks in photoacoustic mesoscopy*, 2021.
* Modifications made here:
  * `computeVoxel_fluid.py` extends the voxel traversal to carry a
    time-dependent contrast concentration per voxel rather than a binary
    occupancy;
  * `vSystem.py` adds coronary- and carotid-specific grammars;
  * unused upstream modules (`preprocessing.py`, `tiff_to_hfd5.py`,
    `visuals.py`, `computeVoxel.py`) were removed to reduce the dependency
    surface; commented-out dead code was deleted.

**Licensing consequence:** because `data_engine/step1_generate_projections.py`
imports `vsystem`, the data engine is a derivative work of GPL-3.0 code. This
repository is therefore distributed under the GPL-3.0 (see `LICENSE`).

## BBDM (`models/bridge.py`)

The Brownian-bridge forward/reverse process follows
[BBDM](https://github.com/xuekt98/BBDM) (Li et al., *BBDM: Image-to-image
Translation with Brownian Bridge Diffusion Models*, CVPR 2023), MIT licensed.
TDBM adds the temporal conditioning, the configurable accelerated sampling
schedule and the vessel-aware contrastive objective.

## Guided Diffusion / Palette (`models/unet.py`, `models/nn.py`)

The attention U-Net trunk follows the widely reused architecture of
[guided-diffusion](https://github.com/openai/guided-diffusion) (OpenAI, MIT) as
adapted in [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
(MIT). The feature-tapping hooks used by the contrastive loss are our addition.

## PatchNCE (`losses.py`)

The fallback patch-NCE formulation follows
[CUT](https://github.com/taesungp/contrastive-unpaired-translation)
(Park et al., ECCV 2020), BSD licensed.

## diffdrr (`data_engine/`)

DRR rendering uses [DiffDRR](https://github.com/eigenvivek/DiffDRR)
(Gopalakrishnan & Golland, 2022), MIT licensed, installed as a dependency rather
than vendored.
