"""Vessel extraction with a trained TDBM model.

TDBM predicts the *contrast-free* counterpart of each input frame.  The vessels
are then recovered as the residual between the input and the prediction, which
preserves the intra-arterial contrast intensity instead of collapsing it into a
binary mask.

Two inference modes are available:

``predict_sequence``
    One forward pass per frame at the model's native resolution.  Fast, and the
    right choice when the vessels occupy most of the field of view.

``predict_sequence_multicrop``
    Runs the model on an ``n x n`` grid of overlapping crops and fuses the
    per-crop residuals with a pixel-wise minimum (residuals are stored inverted,
    so the minimum keeps the strongest vessel response).  This preserves thin
    distal vessels that are lost when a full frame is downsampled, and is the
    setting used for the reported benchmark numbers.

Each mode writes up to three outputs per frame:

``reconstruction/``  the predicted contrast-free frame;
``vessel/``          the inverted residual, i.e. the extracted vessels;
``residual/``        the same residual with 2x contrast boost, which is what
                     ``evaluate.py`` thresholds.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset_xca import ImageDataset
from utils import build_model, load_checkpoint, load_config


def generate_grid_crops(image, crop_size, num_crops):
    """Yield ``num_crops x num_crops`` evenly spaced crops and their offsets.

    Args:
        image: Input tensor ``[B, C, H, W]``.
        crop_size: ``(height, width)`` of each crop.
        num_crops: Number of crops per axis.

    Yields:
        ``(crop_tensor, (top, left))``.
    """
    _, _, height, width = image.shape
    crop_h, crop_w = crop_size

    step_h = (height - crop_h) / (num_crops - 1) if num_crops > 1 else 0
    step_w = (width - crop_w) / (num_crops - 1) if num_crops > 1 else 0

    for i in range(num_crops):
        for j in range(num_crops):
            top = min(int(i * step_h), height - crop_h)
            left = min(int(j * step_w), width - crop_w)
            yield image[:, :, top : top + crop_h, left : left + crop_w], (top, left)


def _residual_from_pair(reconstruction, original, boost=0.5):
    """Convert a (prediction, input) pair of uint8 images into a vessel residual.

    The prediction removes the contrast agent, so ``prediction - input`` is
    positive exactly where vessels were.  The result is inverted so that vessels
    appear dark on a bright background, matching the convention of the
    ground-truth annotations.

    Args:
        reconstruction: Predicted contrast-free frame, uint8.
        original: Input contrast-filled frame, uint8.
        boost: Divisor applied to the difference; ``0.5`` doubles the contrast.
    """
    diff = reconstruction.astype(np.int16) - original.astype(np.int16)
    return (255 - np.clip(diff / boost, 0, 255)).astype(np.uint8)


def _make_loader(datapath, neighbor_num, resize_to):
    transform = transforms.Compose([transforms.Resize((resize_to, resize_to)), transforms.ToTensor()])
    dataset = ImageDataset(image_dir=datapath, transform=transform, n=neighbor_num // 2)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def _save(executor, output_folder, subdir, datapath, image_path, array):
    save_path = os.path.join(output_folder, subdir, os.path.relpath(image_path, datapath))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    executor.submit(cv2.imwrite, save_path, array)


@torch.no_grad()
def predict_sequence(
    model,
    datapath,
    output_folder,
    neighbor_num=3,
    image_size=256,
    device="cuda",
    save_reconstruction=True,
):
    """Run whole-frame inference over every frame under ``datapath``.

    Args:
        model: Trained ``BrownianBridgeModel``.
        datapath: Root directory of input frames (one sub-directory per sequence).
        output_folder: Where to write the outputs.
        neighbor_num: Size of the temporal window; must match the trained model.
        image_size: Resolution the model runs at.
        device: Torch device.
        save_reconstruction: Also write the predicted contrast-free frames.
    """
    os.makedirs(output_folder, exist_ok=True)
    loader = _make_loader(datapath, neighbor_num, image_size)
    model.eval()
    center = neighbor_num // 2

    with ThreadPoolExecutor(max_workers=4) as executor:
        for image, image_path in tqdm(loader, desc="inference"):
            image = image.to(device, non_blocking=True)
            reconstruction = model.sample(
                y=image[:, center, :, :].unsqueeze(1),
                context=image,
                clip_denoised=False,
                progress=False,
            )

            original = image[:, center].squeeze().cpu().numpy()
            reconstruction = reconstruction.squeeze().cpu().numpy()
            original = (original * 255).astype(np.uint8)
            reconstruction = (reconstruction * 255).astype(np.uint8)

            if save_reconstruction:
                _save(executor, output_folder, "reconstruction", datapath, image_path[0], reconstruction)
            _save(executor, output_folder, "vessel", datapath, image_path[0],
                  _residual_from_pair(reconstruction, original, boost=1.0))
            _save(executor, output_folder, "residual", datapath, image_path[0],
                  _residual_from_pair(reconstruction, original, boost=0.5))

    print(f"Inference complete. Results written to {output_folder}")


@torch.no_grad()
def predict_sequence_multicrop(
    model,
    datapath,
    output_folder,
    neighbor_num=3,
    image_size=256,
    crop_size=224,
    num_crops=3,
    resize_to=448,
    device="cuda",
    use_fp16=True,
):
    """Run grid-crop inference and fuse the per-crop residuals.

    Each frame is upsampled to ``resize_to``, split into ``num_crops x num_crops``
    crops of ``crop_size``, and every crop is resized to ``image_size`` before
    being passed through the bridge.  All crops of one frame form a single batch,
    so the cost is one sampling loop per frame regardless of ``num_crops``.

    Args:
        model: Trained ``BrownianBridgeModel``.
        datapath: Root directory of input frames.
        output_folder: Where to write the fused residuals.
        neighbor_num: Size of the temporal window; must match the trained model.
        image_size: Resolution the model runs at.
        crop_size: Side length of each crop, in ``resize_to`` pixels.
        num_crops: Number of crops per axis.
        resize_to: Working resolution the frame is upsampled to before cropping.
        device: Torch device.
        use_fp16: Run the sampling loop under autocast (CUDA only).
    """
    os.makedirs(output_folder, exist_ok=True)
    loader = _make_loader(datapath, neighbor_num, resize_to)
    model.eval()
    center = neighbor_num // 2
    use_fp16 = use_fp16 and device.startswith("cuda")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for image, image_path in tqdm(loader, desc="inference"):
            image = image.to(device, non_blocking=True)
            height, width = image.shape[2], image.shape[3]

            crops, offsets = [], []
            for crop_tensor, offset in generate_grid_crops(image, (crop_size, crop_size), num_crops):
                crops.append(
                    F.interpolate(crop_tensor, size=(image_size, image_size),
                                  mode="bilinear", align_corners=False)
                )
                offsets.append(offset)
            batch = torch.cat(crops, dim=0).contiguous()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                predictions = model.sample(
                    y=batch[:, center, :, :].unsqueeze(1),
                    context=batch,
                    clip_denoised=False,
                    progress=False,
                )

            predictions = predictions.float().cpu()
            batch = batch.cpu()
            originals_np = (batch[:, center].numpy() * 255).astype(np.uint8)
            predictions_np = (predictions.squeeze(1).numpy() * 255).astype(np.uint8)

            # Paste each crop's residual onto a blank (=255, no vessel) canvas so
            # that the pixel-wise minimum below only ever strengthens a response.
            canvases = []
            for (top, left), prediction, original in zip(offsets, predictions_np, originals_np):
                residual = _residual_from_pair(prediction, original)
                canvas = np.full((height, width), 255, dtype=np.uint8)
                canvas[top : top + crop_size, left : left + crop_size] = cv2.resize(
                    residual, (crop_size, crop_size)
                )
                canvases.append(canvas)

            merged = np.min(np.stack(canvases, axis=0), axis=0).astype(np.uint8)
            _save(executor, output_folder, "residual", datapath, image_path[0], merged)

    print(f"Inference complete ({num_crops}x{num_crops} crops). Results written to {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Vessel extraction with a trained TDBM model.")
    parser.add_argument("--config", default="config/tdbm.json", help="Training config used for the checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a TDBM checkpoint (.pth).")
    parser.add_argument("--input", required=True, help="Root directory of input frames; one sub-directory per sequence.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--mode", choices=["full", "multicrop"], default="multicrop",
                        help="Whole-frame inference or grid-crop inference (default).")
    parser.add_argument("--crop-size", type=int, default=224, help="Crop side length, in --resize-to pixels.")
    parser.add_argument("--num-crops", type=int, default=3, help="Crops per axis for --mode multicrop.")
    parser.add_argument("--resize-to", type=int, default=448, help="Working resolution before cropping.")
    parser.add_argument("--sample-step", type=int, default=None,
                        help="Number of reverse diffusion steps K (default: value from the config).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-fp16", action="store_true", help="Disable half-precision sampling.")
    args = parser.parse_args()

    config = load_config(args.config)
    model = build_model(config, args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    if args.sample_step is not None:
        model.set_sample_step(args.sample_step)

    if args.mode == "full":
        predict_sequence(
            model, args.input, args.output,
            neighbor_num=config["neighbor_num"], image_size=config["imgSize"], device=args.device,
        )
    else:
        predict_sequence_multicrop(
            model, args.input, args.output,
            neighbor_num=config["neighbor_num"], image_size=config["imgSize"],
            crop_size=args.crop_size, num_crops=args.num_crops, resize_to=args.resize_to,
            device=args.device, use_fp16=not args.no_fp16,
        )


if __name__ == "__main__":
    main()
