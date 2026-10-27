"""Step 3 of the physics-based XCA data engine: add acquisition noise.

Adds zero-mean, signal-dependent acquisition noise to the composed images,
emulating the quantum noise of real (log-processed) X-ray fluoroscopy.  Because
the model is trained to map a contrast-filled frame to its contrast-free
counterpart, the noise realisation matters: if the background were noise-free the
network could trivially detect vessels by looking for noise.  Three regimes are
therefore sampled per image pair:

* 20% no noise at all;
* 40% the *same* noise realisation shared by both contrast-filled images and the
  background, so that noise carries no information about the vessels;
* 40% an *independent* realisation per image, so that the model also learns to be
  robust to uncorrelated noise between input and target.

Noise model (``--model calibrated``, default)
--------------------------------------------
Displayed XCA frames are log-processed, so the noise is *not* Poisson in the
displayed gray level: fewer photons reach the detector behind dense tissue and
contrast, and the log transform turns that into *larger* noise in *darker*
pixels.  We therefore add zero-mean Gaussian noise whose standard deviation
depends on the local gray level,

    sigma(I) = sigma0 * (128 / clip(I, 96, 255)) ** 0.85          [gray levels]

with ``sigma0`` drawn per image from ``--sigma-range`` (default 1.2-3.6 gray).
The law was fitted on 150 real contrast-free frames (flat-region high-pass
residual binned by gray level: ~2.7-2.9 gray below 130 gray, 2.4 at 128-160,
1.9 at 200, 1.4 at 230); it is flat in the dark range and decreases towards
bright pixels.  In the *shared* regime the noise field is drawn from the clean
background (``image1_ori``) so that it is identical on all four images and
carries no vessel information; in the *independent* regime every image gets its
own field drawn from its own gray levels, which makes contrast-filled pixels
slightly noisier, as in a real detector.

Why the previous implementation was changed
-------------------------------------------
The earlier code added a raw Poisson *sample* ``poisson(I/255 * scale)`` to the
image.  That term has mean ``I * scale / 255``, so it was mostly a brightness
bias of 8-24 % (scale 20-60) with only a small random part; its variance grew
with brightness (the opposite of log-processed X-ray); and in the shared regime
the field was drawn from the *vessel-filled* ``image1``, so the background
targets received a vessel-shaped dip of ``scale/255 * V`` (8-24 % of the vessel
attenuation), i.e. the model was taught to leave part of the vessel in the
"contrast-free" output.  ``--model poisson-legacy`` reproduces the old behaviour
for comparison only.

All other sub-directories (the masks) are copied through unchanged.

Usage::

    python step3_add_poisson_noise.py --input data/synthetic/train_clean \\
        --output data/synthetic/train
"""

import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# Sub-directories holding intensity images; the label directories are copied as-is.
NOISE_SUBDIRS = ("image1", "image1_ori", "image2", "image2_ori")
# The shared, vessel-free background; the shared noise field is drawn from it.
BACKGROUND_SUBDIR = "image1_ori"

# Gray-level dependence of the noise standard deviation (fitted on real frames).
SIGMA_REF_GRAY = 128.0
SIGMA_EXPONENT = 0.85
SIGMA_MIN_GRAY = 96.0


def noise_sigma(img, sigma0):
    """Per-pixel noise standard deviation (gray levels) for the image ``img``."""
    gray = np.clip(img.astype(np.float32), SIGMA_MIN_GRAY, 255.0)
    return sigma0 * (SIGMA_REF_GRAY / gray) ** SIGMA_EXPONENT


def sample_calibrated_noise(img, sigma0, rng):
    """Zero-mean Gaussian noise field with the log-domain gray-level law."""
    return rng.normal(0.0, 1.0, size=img.shape).astype(np.float32) * noise_sigma(img, sigma0)


def sample_poisson_noise(img, scale, rng):
    """Legacy field: a raw Poisson sample with mean ``img / 255 * scale`` (biased)."""
    return rng.poisson(lam=(img / 255.0) * scale).astype(np.float32)


def add_noise(img, noise=None):
    """Add a pre-computed noise field to ``img`` and clip back to uint8."""
    if noise is None:
        return img
    return np.clip(np.rint(img.astype(np.float32) + noise), 0, 255).astype(np.uint8)


def process_dataset(input_dir, output_dir, sigma_range=(1.2, 3.6), scale_range=(20, 60),
                    regime_probs=(0.2, 0.4, 0.4), model="calibrated", seed=None):
    """Apply the three-regime acquisition-noise model to a composed dataset."""
    rng = np.random.default_rng(seed)
    if abs(sum(regime_probs) - 1.0) > 1e-6:
        raise ValueError(f"--regime-probs must sum to 1, got {regime_probs}")
    p_none, p_shared = regime_probs[0], regime_probs[0] + regime_probs[1]

    os.makedirs(output_dir, exist_ok=True)

    # Copy every non-image sub-directory (masks) through unchanged.
    for entry in sorted(os.listdir(input_dir)):
        source = os.path.join(input_dir, entry)
        if os.path.isdir(source) and entry not in NOISE_SUBDIRS:
            shutil.copytree(source, os.path.join(output_dir, entry), dirs_exist_ok=True)
            print(f"Copied {entry}")

    reference_dir = os.path.join(input_dir, BACKGROUND_SUBDIR)
    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(f"{reference_dir} not found: step 2 output expected")
    file_names = sorted(f for f in os.listdir(reference_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    for subdir in NOISE_SUBDIRS:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    def draw(img, level):
        if model == "poisson-legacy":
            return sample_poisson_noise(img, level, rng)
        return sample_calibrated_noise(img, level, rng)

    counts = {"none": 0, "shared": 0, "independent": 0}
    for file_name in tqdm(file_names, desc="adding noise"):
        background = np.array(Image.open(os.path.join(reference_dir, file_name)).convert("L"))

        regime = rng.random()
        level = rng.uniform(*sigma_range) if model == "calibrated" else rng.integers(*scale_range)
        if regime < p_none:
            shared_noise, independent = None, False
            counts["none"] += 1
        elif regime < p_shared:
            # One field for all four images, drawn from the vessel-FREE background:
            # identical noise on input and target, no vessel information in it.
            shared_noise, independent = draw(background, level), False
            counts["shared"] += 1
        else:
            shared_noise, independent = None, True
            counts["independent"] += 1

        for subdir in NOISE_SUBDIRS:
            input_path = os.path.join(input_dir, subdir, file_name)
            if not os.path.exists(input_path):
                continue

            img = np.array(Image.open(input_path).convert("L"))
            if independent:
                # Each image gets its own realisation, drawn from its own gray levels
                # (contrast-filled pixels are darker and therefore slightly noisier).
                noise = draw(img, level)
            else:
                noise = shared_noise
                if noise is not None and noise.shape != img.shape:
                    raise ValueError(f"{subdir}/{file_name}: shape {img.shape} differs from the background {noise.shape}")
            Image.fromarray(add_noise(img, noise)).save(os.path.join(output_dir, subdir, file_name))

    print(f"Done. Noisy dataset written to {output_dir}  (regimes: {counts})")


def main():
    parser = argparse.ArgumentParser(description="Add randomised acquisition noise to a composed dataset.")
    parser.add_argument("--input", required=True, help="Dataset directory produced by step 2.")
    parser.add_argument("--output", required=True, help="Destination directory.")
    parser.add_argument("--model", choices=["calibrated", "poisson-legacy"], default="calibrated",
                        help="calibrated: zero-mean, darker = noisier (default); poisson-legacy: the previous biased model.")
    parser.add_argument("--sigma-range", type=float, nargs=2, default=[1.2, 3.6],
                        help="calibrated model: per-image noise std at 128 gray, in gray levels.")
    parser.add_argument("--scale-range", type=int, nargs=2, default=[20, 60],
                        help="poisson-legacy model: photon-count scale range (lower = noisier).")
    parser.add_argument("--regime-probs", type=float, nargs=3, default=[0.2, 0.4, 0.4],
                        help="Probabilities of the none / shared / independent regimes.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    process_dataset(args.input, args.output, tuple(args.sigma_range), tuple(args.scale_range),
                    tuple(args.regime_probs), args.model, args.seed)


if __name__ == "__main__":
    main()
