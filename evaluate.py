"""Binary vessel-segmentation metrics for TDBM residual outputs.

``inference.py`` writes grayscale residuals in which vessels are *dark*.  This
script inverts them, applies a fixed threshold, and compares the result against
the ground-truth annotations, reporting Dice, sensitivity (recall), specificity
and accuracy.

Expected layout::

    <gt_root>/<sequence>/<one annotation image>
    <pred_root>/<sequence>/<one residual per frame>

Only the frames listed in ``--frames`` are used, matching the benchmark protocol
(the public XCA30 set annotates the third frame of every sequence, i.e. index 2).
When several frames are selected, their residuals are fused with a pixel-wise
minimum before thresholding.
"""

import argparse
import csv
import os
from glob import glob

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from tqdm import tqdm

METRIC_NAMES = ("dice", "recall", "specificity", "accuracy")


def evaluate_segmentation(
    gt_root,
    pred_root,
    frame_indices,
    binary_save_root=None,
    threshold=20,
    gt_threshold=10,
):
    """Compute segmentation metrics over every sequence in ``gt_root``.

    Args:
        gt_root: Directory with one sub-directory per sequence, each holding a
            single ground-truth annotation image.
        pred_root: Matching directory of predicted residuals.
        frame_indices: Indices of the frames to score within each sequence.
        binary_save_root: If set, write the thresholded mask, the fused
            prediction and the ground truth here for visual inspection.
        threshold: Intensity threshold applied to the inverted residual.
        gt_threshold: Intensity above which a ground-truth pixel counts as vessel.

    Returns:
        ``(mean_metrics, std_metrics, per_sequence_metrics)``.
    """
    sequences = sorted(os.listdir(gt_root))
    metrics = []

    for sequence in tqdm(sequences, desc="evaluating"):
        gt_dir = os.path.join(gt_root, sequence)
        pred_dir = os.path.join(pred_root, sequence)

        gt_images = sorted(glob(os.path.join(gt_dir, "*.png")) + glob(os.path.join(gt_dir, "*.jpg")))
        if len(gt_images) != 1:
            print(f"Warning: {gt_dir} should contain exactly one annotation image, found {len(gt_images)}.")
            continue
        gt = (cv2.imread(gt_images[0], cv2.IMREAD_GRAYSCALE) > gt_threshold).astype(np.uint8)

        pred_images = sorted(glob(os.path.join(pred_dir, "*.png")) + glob(os.path.join(pred_dir, "*.jpg")))
        selected = []
        for index in frame_indices:
            if index < len(pred_images):
                selected.append(cv2.imread(pred_images[index], cv2.IMREAD_GRAYSCALE))
            else:
                print(f"Frame index {index} out of range in {pred_dir} ({len(pred_images)} frames).")
        if not selected:
            continue

        # Vessels are dark in the residual, so the minimum keeps the strongest
        # response across frames; invert to get a bright-vessel probability map.
        fused = np.clip(255 - np.min(selected, axis=0), 0, 255).astype(np.uint8)
        _, pred_bin = cv2.threshold(fused, threshold, 1, cv2.THRESH_BINARY)

        if pred_bin.shape != gt.shape:
            pred_bin = cv2.resize(pred_bin, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        if binary_save_root:
            save_dir = os.path.join(binary_save_root, sequence)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f"{sequence}.png"), pred_bin * 255)
            cv2.imwrite(os.path.join(save_dir, f"{sequence}_pred.png"), fused)
            cv2.imwrite(os.path.join(save_dir, f"{sequence}_gt.png"), gt * 255)

        gt_flat, pred_flat = gt.flatten(), pred_bin.flatten()
        tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
        metrics.append(
            {
                "sequence": sequence,
                "dice": f1_score(gt_flat, pred_flat),
                "recall": recall_score(gt_flat, pred_flat),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
            }
        )

    if not metrics:
        raise RuntimeError(f"No sequence could be evaluated (gt={gt_root}, pred={pred_root}).")

    mean = {k: float(np.mean([m[k] for m in metrics])) for k in METRIC_NAMES}
    std = {k: float(np.std([m[k] for m in metrics])) for k in METRIC_NAMES}
    return mean, std, metrics


def write_report(mean, std, metrics, output_dir):
    """Write ``average_metrics.txt`` and ``per_sequence_metrics.csv``."""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "average_metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Average Metrics:\n")
        handle.write("=" * 50 + "\n")
        for name in METRIC_NAMES:
            line = f"{name}: {mean[name]:.4f} ± {std[name]:.4f}\n"
            handle.write(line)
            print(line.strip())

    csv_path = os.path.join(output_dir, "per_sequence_metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\nMetrics written to {summary_path} and {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Score TDBM residual outputs against ground-truth masks.")
    parser.add_argument("--gt", required=True, help="Ground-truth root; one sub-directory per sequence.")
    parser.add_argument("--pred", required=True, help="Prediction root, e.g. <output>/residual.")
    parser.add_argument("--frames", type=int, nargs="+", default=[2],
                        help="Frame indices to score (default: 2, the annotated frame of XCA30).")
    parser.add_argument("--threshold", type=int, default=20, help="Binarisation threshold (default: 20).")
    parser.add_argument("--save-binary", default=None, help="Directory for thresholded masks and overlays.")
    parser.add_argument("--report-dir", default=None,
                        help="Where to write the metric report (default: parent of --save-binary, else --pred).")
    args = parser.parse_args()

    mean, std, metrics = evaluate_segmentation(
        args.gt, args.pred, args.frames,
        binary_save_root=args.save_binary, threshold=args.threshold,
    )

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = os.path.dirname(args.save_binary) if args.save_binary else args.pred
    write_report(mean, std, metrics, report_dir)


if __name__ == "__main__":
    main()
