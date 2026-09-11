# Dataset layouts

Every path below is set in `config/tdbm.json`. Nothing in this repository ships
imaging data; `.gitignore` excludes `data/` and `datasets/` so that patient data
cannot be committed by accident.

---

## 1. Synthetic training pairs — `synthetic_data_root`

Produced by `data_engine/`. One file per sample, with the *same* file name across
all sub-directories:

```
<synthetic_data_root>/
├── image1/00000.png        background + vessel tree A   (contrast-filled input)
├── image2/00000.png        background + vessel tree B   (contrast-filled input)
├── label/00000.png         binary mask of tree A
├── label2/00000.png        binary mask of tree B
├── image1_ori/00000.png    the shared contrast-free background (the target)
├── image2_ori/00000.png    the shared contrast-free background
├── label_ori/00000.png     soft vessel intensity map of tree A
└── label2_ori/00000.png    soft vessel intensity map of tree B
```

`image1` and `image2` share a background and differ only in vasculature — this is
what the vessel-aware contrastive loss needs. `image1_ori` is the bridge target
`x0`; `label`/`label2` define the vessel region `Omega_v`.

The temporal window is *synthesised on the fly* by `PairedDataset`: each single
image is expanded into `neighbor_num` frames with random erasing and per-frame
elastic warping, so no multi-frame data is stored on disk.

`image1_ori` and `image2_ori` are identical by construction; both are written so
that the layout stays symmetric between the two views.

---

## 2. Real XCA sequences — `real_data_root`

Used only for the annealed real-data term (`use_real_data: true`). For each
sequence, the frames acquired *after* contrast injection are the inputs and one
frame acquired *before* injection is the contrast-free reference.

```
<real_data_root>/
├── image/
│   ├── case_001/           contrast-filled frames, sorted by file name
│   │   ├── 000.png
│   │   └── ...
│   └── case_002/
└── label/
    ├── case_001/
    │   └── 000.png         a single contrast-free reference frame
    └── case_002/
```

`ImageSequenceDataset` takes the `neighbor_num` frames centred on the middle of
each sequence, and any sequence with fewer frames than that is skipped with a
warning. These pairs are not perfectly registered — cardiac and respiratory
motion means the reference frame does not align with the inputs — which is
exactly why this term is annealed towards zero during training.

---

## 3. Evaluation sets — `val_image_root` / `val_gt_root`

One sub-directory per sequence on both sides, with matching names. The
ground-truth directory holds exactly one annotation image per sequence.

```
<val_image_root>/            <val_gt_root>/
├── volume_00/               ├── volume_00/
│   ├── 000.png              │   └── volume_00.png    (annotation of frame 2)
│   ├── 001.png              ├── volume_01/
│   ├── 002.png   <-- annotated frame (index 2)
│   └── 003.png
└── volume_01/
```

`val_frames` (default `[2]`) selects which frames are scored. The public
[XCA30](https://github.com/Binjie-Qin/SVS-net) benchmark annotates the third
frame of each 4-frame sequence, hence index `2`.

---

## 4. Inference input

`inference.py --input` takes any directory tree; each sub-directory is treated as
an independent sequence and temporal neighbours are only drawn from within it.
Sequence boundaries are padded by repeating the first/last frame. Supported
extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`. Images are read as
grayscale and resized to the model's working resolution.

The output mirrors the input tree, so `results/<sequence>/<frame>.png` lines up
one-to-one with `input/<sequence>/<frame>.png`.

---

## 5. Zero-shot evaluation on other modalities

The retinal (DRIVE) and fluorescence-microscopy (VessMap) experiments use the
same interface: place each image in its own sub-directory (a one-frame
"sequence"), run `inference.py --mode full`, and score with `evaluate.py
--frames 0`. No fine-tuning or modality-specific configuration is applied.
