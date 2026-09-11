"""Train the Temporal Diffusion Bridge Model (TDBM).

The objective combines three terms (Eq. 8 of the paper)::

    L_total = L_sup^syn + alpha * L_vcl + lambda(t) * L_sup^real

``L_sup^syn``
    Brownian-bridge reconstruction loss on the synthetic pairs.  Each sample
    provides two contrast-filled views that share a background, and both are
    mapped back to that background; the two losses are averaged.

``L_vcl``
    Vessel-aware contrastive loss between the intermediate U-Net features of the
    two views (see ``losses.py``).

``L_sup^real``
    The same reconstruction loss on real XCA sequences, where the first frame of
    a sequence serves as the contrast-free reference.  Real pairs are not
    perfectly aligned, so this term is annealed towards zero over training.

Usage::

    python train.py --config config/tdbm.json
    python train.py --config config/tdbm.json --resume outputs/TDBM/model_latest.pth
"""

import argparse
import itertools
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_xca import ImageSequenceDataset, PairedDataset
from evaluate import evaluate_segmentation
from inference import predict_sequence_multicrop
from losses import VesselAwareContrastiveLoss
from utils import (EMA, build_model, clone_for_ema, linear_decay, load_config,
                   set_seed, setup_performance)

# Feature levels tapped for the contrastive loss.  Level 14 is the deepest
# encoder block; the decoder levels are excluded because their features are
# dominated by the reconstruction target rather than by vessel appearance.
VCL_FEATURE_LEVELS = (14,)


def get_dataloaders(config):
    """Build the synthetic-pair loader and (optionally) the real-sequence loader."""
    synthetic_root = config["synthetic_data_root"]
    dataset = PairedDataset(
        synthetic_root,
        transform=True,
        channel=config["neighbor_num"],
        size=(config["imgSize"], config["imgSize"]),
        overall_aug=config["overall_aug"],
    )
    generator = torch.Generator()
    generator.manual_seed(42)
    synthetic_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        # NOTE: the original research runs used shuffle=False, which means only the
        # first `batch_size * num_iterations_per_epoch` samples of the synthetic set
        # were ever visited. Shuffling is enabled by default here so that the full
        # corpus is used; set "shuffle": false to reproduce the original behaviour.
        shuffle=config.get("shuffle", True),
        drop_last=True,
        generator=generator,
        num_workers=config.get("num_workers", 1),
        pin_memory=True,
    )

    real_loader = None
    if config.get("use_real_data", False):
        real_dataset = ImageSequenceDataset(
            data_root=config["real_data_root"],
            num_images=config["neighbor_num"],
            image_size=(config["imgSize"], config["imgSize"]),
            mode="train",
        )
        real_loader = DataLoader(
            real_dataset,
            batch_size=config.get("real_batch_size", 4),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    return synthetic_loader, real_loader


def collect_features(model, levels=VCL_FEATURE_LEVELS):
    """Return the cached U-Net features for the requested levels."""
    saved = model.denoise_fn.get_saved_features()
    missing = [level for level in levels if saved.get(level) is None]
    if missing:
        raise RuntimeError(
            f"U-Net did not expose feature level(s) {missing}; available: {sorted(saved)}. "
            "Check res_blocks/channel_mults in the config."
        )
    return [saved[level] for level in levels]


def save_checkpoint(path, epoch, model, optimizer, scheduler, loss, lr):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "lr": lr,
        },
        path,
    )


@torch.no_grad()
def validate(model, config, save_dir, epoch):
    """Run grid-crop inference on the validation set and return the mean Dice."""
    output_folder = os.path.join(save_dir, "validation", f"epoch_{epoch + 1}")
    predict_sequence_multicrop(
        model,
        datapath=config["val_image_root"],
        output_folder=output_folder,
        neighbor_num=config["neighbor_num"],
        image_size=config["imgSize"],
        crop_size=config.get("val_crop_size", 280),
        num_crops=config.get("val_num_crops", 3),
        resize_to=config.get("val_resize_to", 448),
        device=config["device"],
    )
    mean, _, _ = evaluate_segmentation(
        config["val_gt_root"],
        os.path.join(output_folder, "residual"),
        config.get("val_frames", [2]),
        binary_save_root=os.path.join(output_folder, "binary"),
    )
    print(f"[epoch {epoch + 1}] validation Dice: {mean['dice']:.4f}")
    return mean["dice"]


def train(config, save_dir, resume=None):
    """Train TDBM according to ``config``, writing checkpoints into ``save_dir``."""
    device = config["device"]
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(config, device)
    model_ema = clone_for_ema(model)
    ema = EMA(beta=config.get("ema_beta", 0.9999))

    vcl_loss = VesselAwareContrastiveLoss(
        num_anchors=config["vcl_num_anchors"],
        patch_size=config["vcl_patch_size"],
        num_negatives=config["vcl_num_negatives"],
        temperature=config["vcl_temperature"],
    )

    synthetic_loader, real_loader = get_dataloaders(config)
    real_iter = iter(itertools.cycle(real_loader)) if real_loader is not None else None

    epochs = config["num_epochs"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["initial_lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model_ema.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from '{resume}' at epoch {start_epoch}")

    center = config["neighbor_num"] // 2
    max_iterations = config["num_iterations_per_epoch"]
    best_dice = 0.0
    total_start = time.time()
    print("Start training...\n")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        running_loss, num_batches = 0.0, 0

        # lambda(t): weight of the real-data term, annealed as training proceeds.
        real_weight = linear_decay(
            current_epoch=epoch,
            start_weight=config.get("real_weight_start", 0.5),
            end_weight=config.get("real_weight_end", 0.25),
            total_epochs=epochs,
            start_epoch=config.get("real_weight_start_epoch", 2),
        )

        progress_bar = tqdm(synthetic_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for i, batch in enumerate(progress_bar):
            if i >= max_iterations:
                break

            view1 = batch["image1"].to(device, non_blocking=True)
            view2 = batch["image2"].to(device, non_blocking=True)
            background = batch["image1_ori"].to(device, non_blocking=True)

            vessel1 = batch["label1"].to(device, non_blocking=True) > 0
            vessel2 = batch["label2"].to(device, non_blocking=True) > 0
            vessel_mask = vessel1 | vessel2

            optimizer.zero_grad()

            # Both views share a timestep and a noise draw so that the two feature
            # maps differ only by their vasculature, which is what VCL assumes.
            batch_size = background.shape[0]
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(background)

            loss1, _ = model.p_losses(
                x0=background, y=view1[:, center, :, :].unsqueeze(1), t=t, noise=noise, context=view1
            )
            features1 = collect_features(model)

            loss2, _ = model.p_losses(
                x0=background, y=view2[:, center, :, :].unsqueeze(1), t=t, noise=noise, context=view2
            )
            features2 = collect_features(model)

            loss_vcl = torch.zeros((), device=device)
            if config["use_vcl"]:
                loss_vcl = vcl_loss(features1, features2, vessel_mask)

            loss_real = torch.zeros((), device=device)
            if real_iter is not None:
                real_batch = next(real_iter)
                real_input = real_batch["image"].to(device, non_blocking=True)
                real_background = real_batch["label"].to(device, non_blocking=True)
                loss_real, _ = model(
                    x=real_background,
                    y=real_input[:, center, :, :].unsqueeze(1),
                    context=real_input,
                )

            loss = (
                0.5 * loss1
                + 0.5 * loss2
                + config["vcl_weight"] * loss_vcl
                + real_weight * loss_real
            )

            loss.backward()
            optimizer.step()
            ema.update_model_average(model_ema, model)

            running_loss += loss.item()
            num_batches += 1
            if i % 10 == 0:
                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "vcl": f"{float(loss_vcl):.4f}"}
                )

        avg_loss = running_loss / max(num_batches, 1)
        scheduler.step()
        print(
            f"Epoch [{epoch + 1}/{epochs}] | Avg Loss: {avg_loss:.4f} | "
            f"lambda: {real_weight:.3f} | Time: {time.time() - epoch_start:.2f}s"
        )

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        save_checkpoint(
            os.path.join(save_dir, "model_latest.pth"),
            epoch, model_ema, optimizer, scheduler, avg_loss, config["initial_lr"],
        )

        if (epoch + 1) % config["test_every"] == 0:
            dice = validate(model_ema, config, save_dir, epoch)
            if dice > best_dice:
                best_dice = dice
                print(f"New best model at epoch {epoch + 1} (Dice {dice:.4f}); saving.")
                save_checkpoint(
                    os.path.join(save_dir, "model_best.pth"),
                    epoch, model_ema, optimizer, scheduler, avg_loss, config["initial_lr"],
                )
            model.train()

    print(f"\nTraining complete in {(time.time() - total_start) / 60:.2f} minutes. "
          f"Best validation Dice: {best_dice:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Train TDBM on physics-based synthetic XCA data.")
    parser.add_argument("--config", default="config/tdbm.json", help="Path to a JSON config file.")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: outputs/<experiment_name>).")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume training from.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--seed", type=int, default=10, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_performance()

    config = load_config(args.config)
    config["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.output or os.path.join("outputs", config["experiment_name"])

    train(config, save_dir, resume=args.resume)


if __name__ == "__main__":
    main()
