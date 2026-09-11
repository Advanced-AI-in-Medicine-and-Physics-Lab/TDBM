"""Step 2 of the physics-based XCA data engine: compose paired training images.

Takes the contrast-only projections rendered by ``step1_generate_projections.py``
and fuses them with real contrast-free fluoroscopic backgrounds::

    I_t = B - alpha * V_t

where ``B`` is a background built by MixUp-ing two randomly drawn real frames
(with random contrast/gamma jitter) and ``V_t`` is a randomly scaled, shifted and
rotated vessel projection.

Every sample writes **two** contrast-filled images that share the *same*
background but carry *different* vessel trees.  That background invariance is
what the vessel-aware contrastive loss exploits: the only systematic difference
between the two images is the vasculature.

Output layout (consumed by ``PairedDataset``)::

    <output>/image1       background + vessel tree A
    <output>/image2       background + vessel tree B
    <output>/label        binary mask of tree A
    <output>/label2       binary mask of tree B
    <output>/image1_ori   the shared contrast-free background
    <output>/image2_ori   the shared contrast-free background
    <output>/label_ori    soft vessel intensity map of tree A
    <output>/label2_ori   soft vessel intensity map of tree B

Poisson noise is added afterwards by ``step3_add_poisson_noise.py``, which keeps
the noise realisation consistent across the paired images.

Usage::

    python step2_compose_xca.py --vessels output/projections \\
        --backgrounds /path/to/contrast_free_frames --output data/synthetic/train \\
        --num-samples 50000
"""

import argparse
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
OUTPUT_SUBDIRS = (
    "image1", "image2", "label", "label2",
    "image1_ori", "image2_ori", "label_ori", "label2_ori",
)


def adjust_contrast_gamma(img, contrast_range=(0.8, 1.2), gamma_range=(0.8, 1.2)):
    """Apply random contrast and gamma jitter to a grayscale image.

    Contrast: ``I' = a * (I - mean) + mean``.  Gamma: ``I' = 255 * (I / 255) ** g``.
    """
    contrast_factor = random.uniform(*contrast_range)
    gamma_value = random.uniform(*gamma_range)

    mean_intensity = np.mean(img)
    adjusted = cv2.convertScaleAbs(
        img, alpha=contrast_factor, beta=(1 - contrast_factor) * mean_intensity
    )
    corrected = np.power(adjusted / 255.0, gamma_value) * 255
    return np.clip(corrected, 0, 255).astype(np.uint8)


def augment_vessel_projection(projection, mask_threshold=20):
    """Randomly rescale, shift and rotate a vessel projection.

    Zooming out pads the projection at a random offset (so the tree is not always
    centred); zooming in centre-crops it.  The binary mask is derived after the
    geometric transform so that it stays exactly aligned with the vessels.

    Returns:
        ``(vessel_image, binary_mask)``.
    """
    target_height, target_width = projection.shape[:2]
    scale_factor = random.uniform(0.8, 1.3)
    scaled = cv2.resize(
        projection, (int(target_width * scale_factor), int(target_height * scale_factor))
    )

    if scale_factor <= 1.0:
        padded = np.zeros_like(projection)
        y_offset = random.randint(0, padded.shape[0] - scaled.shape[0])
        x_offset = random.randint(0, padded.shape[1] - scaled.shape[1])
        padded[y_offset : y_offset + scaled.shape[0], x_offset : x_offset + scaled.shape[1]] = scaled
        result = padded
    else:
        start_y = (scaled.shape[0] - target_height) // 2
        start_x = (scaled.shape[1] - target_width) // 2
        result = scaled[start_y : start_y + target_height, start_x : start_x + target_width]

    _, mask = cv2.threshold(result, mask_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    angle = random.choice([0, 90, 180, 270])
    if angle:
        rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[angle]
        result = cv2.rotate(result, rotate_code)
        mask = cv2.rotate(mask, rotate_code)

    return np.clip(result, 0, 255).astype(np.uint8), mask


def random_resize_or_crop(image, size):
    """Bring ``image`` to ``size x size`` by either resizing or random cropping."""
    if random.choice(["resize", "crop"]) == "resize":
        return cv2.resize(image, (size, size))

    height, width = image.shape[:2]
    if height < size or width < size:
        image = cv2.resize(image, (max(width, size), max(height, size)))
        height, width = image.shape[:2]

    x_start = random.randint(0, width - size)
    y_start = random.randint(0, height - size)
    return image[y_start : y_start + size, x_start : x_start + size]


def create_mixed_background(background_files, size=300):
    """Build a contrast-free background by MixUp-ing two real frames.

    Half the time the first frame is returned unmixed, so the model also sees
    unmodified real background statistics.
    """
    first, second = random.sample(background_files, 2)
    bg1 = random_resize_or_crop(cv2.imread(first, cv2.IMREAD_GRAYSCALE), size)
    bg2 = random_resize_or_crop(cv2.imread(second, cv2.IMREAD_GRAYSCALE), size)

    bg1 = adjust_contrast_gamma(bg1)
    bg2 = adjust_contrast_gamma(bg2)

    if random.random() < 0.5:
        return bg1

    alpha = random.uniform(0, 1)
    return (bg1.astype(np.float32) * alpha + bg2.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def compose_pair(vessel_path_a, vessel_path_b, background_files, output_dir, index,
                 background_size=300, ratio_range=(0.20, 0.36)):
    """Fuse two vessel projections onto one shared background and save all roles.

    Args:
        vessel_path_a: Projection used for ``image1``.
        vessel_path_b: Projection used for ``image2``.
        background_files: Pool of real contrast-free frames.
        output_dir: Destination root.
        index: Sample index; determines the file name.
        background_size: Side length of the generated images.
        ratio_range: Range of ``alpha``, controlling vessel visibility.
    """
    background = create_mixed_background(background_files, background_size)
    height, width = background.shape[:2]

    vessel_a = cv2.resize(cv2.imread(vessel_path_a, cv2.IMREAD_GRAYSCALE), (width, height))
    vessel_b = cv2.resize(cv2.imread(vessel_path_b, cv2.IMREAD_GRAYSCALE), (width, height))

    vessel_a, mask_a = augment_vessel_projection(vessel_a)
    vessel_b, mask_b = augment_vessel_projection(vessel_b)

    # I_t = B - alpha * V_t: contrast attenuates the X-ray beam, so vessels darken
    # the background rather than adding to it.
    ratio = random.uniform(*ratio_range)
    fused_a = np.clip(background.astype(np.int16) - (vessel_a * ratio).astype(np.int16), 0, 255)
    fused_b = np.clip(background.astype(np.int16) - (vessel_b * ratio).astype(np.int16), 0, 255)

    file_name = f"{index:05d}.png"
    for subdir in OUTPUT_SUBDIRS:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    def write(subdir, array):
        cv2.imwrite(os.path.join(output_dir, subdir, file_name), array.astype(np.uint8))

    write("image1", fused_a)
    write("image2", fused_b)
    write("label", mask_a)
    write("label2", mask_b)
    write("image1_ori", background)
    write("image2_ori", background)
    write("label_ori", vessel_a * ratio)
    write("label2_ori", vessel_b * ratio)


def list_images(directory):
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not files:
        raise ValueError(f"No images found in {directory}")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Fuse synthetic vessel projections with real contrast-free backgrounds."
    )
    parser.add_argument("--vessels", required=True, help="Directory of vessel projections from step 1.")
    parser.add_argument("--backgrounds", required=True,
                        help="Directory of real XCA frames acquired before contrast injection.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of pairs to generate (default: one per vessel projection).")
    parser.add_argument("--size", type=int, default=300, help="Output image size.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    vessel_files = list_images(args.vessels)
    background_files = list_images(args.backgrounds)
    random.shuffle(vessel_files)

    num_samples = args.num_samples or len(vessel_files)
    print(f"Composing {num_samples} pairs from {len(vessel_files)} projections "
          f"and {len(background_files)} backgrounds.")

    for index in tqdm(range(num_samples), desc="composing"):
        path_a = vessel_files[index % len(vessel_files)]
        # The second tree is drawn from a nearby index so that paired samples come
        # from projections of similar generation parameters.
        offset = random.randint(1, 2)
        path_b = vessel_files[(index + offset) % len(vessel_files)]
        compose_pair(path_a, path_b, background_files, args.output, index, args.size)

    print(f"Done. Dataset written to {args.output}")


if __name__ == "__main__":
    main()
