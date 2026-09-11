"""Step 3 of the physics-based XCA data engine: add acquisition noise.

Adds signal-dependent Poisson noise to the composed images, emulating the quantum
noise of real X-ray acquisition.  Because the model is trained to map a
contrast-filled frame to its contrast-free counterpart, the noise realisation
matters: if the background were noise-free the network could trivially detect
vessels by looking for noise.  Three regimes are therefore sampled per image
pair:

* 20% no noise at all;
* 40% the *same* noise realisation shared by both contrast-filled images and the
  background, so that noise carries no information about the vessels;
* 40% an *independent* realisation per image, so that the model also learns to be
  robust to uncorrelated noise between input and target.

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


def sample_poisson_noise(img, scale):
    """Draw a signal-dependent Poisson noise field for ``img``.

    Args:
        img: Grayscale image, uint8.
        scale: Photon-count scale; lower means noisier.
    """
    return np.random.poisson(lam=(img / 255.0) * scale)


def add_poisson_noise(img, poisson_noise=None):
    """Add a pre-computed noise field to ``img`` and clip back to uint8."""
    if poisson_noise is None:
        return img
    return np.clip(img + poisson_noise, 0, 255).astype(np.uint8)


def process_dataset(input_dir, output_dir, scale_range=(20, 60), seed=None):
    """Apply the three-regime Poisson noise model to a composed dataset."""
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Copy every non-image sub-directory (masks) through unchanged.
    for entry in sorted(os.listdir(input_dir)):
        source = os.path.join(input_dir, entry)
        if os.path.isdir(source) and entry not in NOISE_SUBDIRS:
            shutil.copytree(source, os.path.join(output_dir, entry), dirs_exist_ok=True)
            print(f"Copied {entry}")

    reference_dir = os.path.join(input_dir, NOISE_SUBDIRS[0])
    file_names = sorted(f for f in os.listdir(reference_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    for subdir in NOISE_SUBDIRS:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for file_name in tqdm(file_names, desc="adding noise"):
        reference = np.array(Image.open(os.path.join(reference_dir, file_name)).convert("L"))

        regime = np.random.rand()
        scale = np.random.randint(*scale_range)
        if regime < 0.2:
            shared_noise, independent = None, False
        elif regime < 0.6:
            shared_noise, independent = sample_poisson_noise(reference, scale), False
        else:
            shared_noise, independent = None, True

        for subdir in NOISE_SUBDIRS:
            input_path = os.path.join(input_dir, subdir, file_name)
            if not os.path.exists(input_path):
                continue

            img = np.array(Image.open(input_path).convert("L"))
            noise = sample_poisson_noise(img, scale) if independent else shared_noise
            noisy = add_poisson_noise(img, noise)
            Image.fromarray(noisy).save(os.path.join(output_dir, subdir, file_name))

    print(f"Done. Noisy dataset written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Add randomised Poisson noise to a composed dataset.")
    parser.add_argument("--input", required=True, help="Dataset directory produced by step 2.")
    parser.add_argument("--output", required=True, help="Destination directory.")
    parser.add_argument("--scale-range", type=int, nargs=2, default=[20, 60],
                        help="Photon-count scale range; lower values give noisier images.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    process_dataset(args.input, args.output, tuple(args.scale_range), args.seed)


if __name__ == "__main__":
    main()
