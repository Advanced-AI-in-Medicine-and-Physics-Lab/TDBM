"""Step 1 of the physics-based XCA data engine: vessel trees -> DRR projections.

For every synthetic sample this script

1. grows a 3-D arterial tree with a stochastic Lindenmayer system whose
   bifurcation angles and diameter tapering follow Murray's law (``vsystem``);
2. simulates contrast propagation along that tree, giving every voxel an arrival
   time from the cumulative path integral of segment velocities and a
   gamma-variate enhancement profile (``util.bolus_simulation``);
3. rasterises the 4-D vascular field into a volume via voxel traversal; and
4. renders the volume to a 2-D detector with a differentiable digitally
   reconstructed radiograph (DRR).

The output is one contrast-only projection per timestep, which
``step2_compose_xca.py`` then fuses with real contrast-free backgrounds.

Usage::

    python step1_generate_projections.py --reference-volume /path/to/ct.nii.gz \\
        --output-dir output --num-samples 2000
"""

import argparse
import math
import os
import random
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from diffdrr.data import read
from diffdrr.drr import DRR
from PIL import Image, ImageEnhance

from util.bolus_simulation import bolus_injection
from util.helper import save_numpy_to_nii
from vsystem.analyseGrammar import branching_turtle_to_coords
from vsystem.computeVoxel_fluid import process_network_fluid, resize_network
from vsystem.libGenerator import setProperties
from vsystem.utils import bezier_interpolation
from vsystem.vSystem import F

# Attenuation assigned to vessel-free voxels (air, in Hounsfield units) and the
# scale applied to the simulated contrast concentration.
AIR_HU = -1000
CONTRAST_SCALE = 1000


def read_nii_header(file_path):
    """Return the NIfTI header of ``file_path``, or ``None`` if it cannot be read."""
    try:
        return nib.load(file_path).header
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        print(f"Error reading file {file_path}: {exc}")
        return None


def drr_process(
    volume,
    rotate,
    translate,
    meta_csv=None,
    device="cuda",
    bone_multi=1.0,
    orientation="AP",
    default_sdd=1000,
    height=512,
):
    """Render one DRR of ``volume`` for the given C-arm pose.

    Args:
        volume: Path to the NIfTI volume to project.
        rotate: Euler angles (ZXY convention), shape ``[1, 3]``.
        translate: Source-detector translation, shape ``[1, 3]``.
        meta_csv: Optional CSV providing a per-volume ``DistanceSourceToDetector``.
        device: Torch device for the renderer.
        bone_multi: Bone attenuation multiplier.
        orientation: Patient orientation passed to ``diffdrr``.
        default_sdd: Source-to-detector distance used when the CSV has none.
        height: Detector size in pixels.

    Returns:
        The rendered projection as a 2-D tensor.
    """
    sdd = float("nan")
    if meta_csv is not None:
        table = pd.read_csv(meta_csv)
        row = table[table["VolumeName"] == os.path.basename(volume)]
        if len(row):
            sdd = float(row["DistanceSourceToDetector"].iloc[0])
    if math.isnan(sdd):
        sdd = default_sdd

    subject = read(volume, orientation=orientation, bone_attenuation_multiplier=bone_multi)
    drr = DRR(subject, sdd=sdd, height=height, delx=1, renderer="trilinear").to(device)

    img = drr(
        torch.tensor(rotate, device=device),
        torch.tensor(translate, device=device),
        parameterization="euler_angles",
        convention="ZXY",
        n_points=1000,
    )
    del subject, drr
    return img[0][0]


def render_projection(volume_path, output_folder, config, device="cuda"):
    """Render ``volume_path`` from every configured viewpoint and save as PNG.

    Randomised rotations emulate the varying fluoroscopic viewpoints and patient
    positioning seen across acquisitions.
    """
    os.makedirs(output_folder, exist_ok=True)
    sdd = config["sdd"]
    translate = [[0.0, sdd - config["translation"], 0.0]]

    for rotation in config["rotations"]:
        with torch.no_grad():
            projection = drr_process(
                volume=volume_path,
                rotate=[rotation],
                translate=translate,
                device=device,
                bone_multi=config.get("bone_attenuation_multiplier", 1.5),
                orientation=config.get("orientation", "AP"),
                default_sdd=sdd,
                height=config.get("detector_height", 512),
            )

        normalized = (
            255 * (projection - projection.min()) / (projection.max() - projection.min())
        ).byte()
        image = Image.fromarray(normalized.cpu().numpy(), "L")
        image = ImageEnhance.Contrast(image).enhance(config.get("contrast_enhance", 3.0))

        file_name = os.path.basename(volume_path)[:-7] + str(rotation) + ".png"
        image.save(os.path.join(output_folder, file_name))

        del projection
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def sample_tree_properties(config):
    """Draw one set of L-system and contrast-dynamics parameters."""
    ranges = config["vessel_generation"]
    flow = config["contrast_dynamics"]
    return {
        "d0": np.random.normal(
            np.random.uniform(ranges["d0_mean_min"], ranges["d0_mean_max"]), ranges["d0_std"]
        ),
        "niter": random.randint(*ranges["niter_range"]),
        "properties": {
            "k": np.random.uniform(*ranges["k_range"]),            # branching multiplier
            "epsilon": np.random.uniform(*ranges["epsilon_range"]),  # tortuosity
            "randmarg": np.random.uniform(*ranges["randmarg_range"]),  # irregularity
            "sigma": np.random.uniform(*ranges["sigma_range"]),      # noise magnitude
            "d": np.random.uniform(*ranges["d_ratio_range"]),        # parent:child diameter ratio
            "stochparams": True,
        },
        "alpha": np.random.uniform(*flow["alpha_range"]),  # gamma-variate shape
        "beta": np.random.uniform(*flow["beta_range"]),    # gamma-variate clearance
        "v_base": np.random.uniform(*flow["v_base_range"]),  # base flow velocity
        "p_flow": flow.get("p_flow", 1.0),                   # flow-to-diameter exponent
    }


def generate_sample(index, config, tissue_volume, header, affine, paths, device="cuda"):
    """Generate one vessel tree and render its projection at every timestep."""
    params = sample_tree_properties(config)
    start = time.time()
    print(
        f"[{index}] vessel: niter={params['niter']}, d0={params['d0']:.2f}, "
        f"alpha={params['alpha']:.2f}, beta={params['beta']:.2f}, "
        f"v_base={params['v_base']:.2f}"
    )

    setProperties(params["properties"])
    turtle_program = F(params["niter"], params["d0"])
    coords = branching_turtle_to_coords(turtle_program, params["d0"])
    network = bezier_interpolation(coords)

    dmin = np.nanmin(network, axis=1) * 1.1
    dmax = np.nanmax(network, axis=1) * 1.1

    timing = config["contrast_dynamics"]
    for t in range(timing["t_start"], timing["t_end"], timing["t_step"]):
        volume_name = os.path.join(
            paths["volumes"],
            f"Lnet_d{params['d0']:.2f}_iter{params['niter']}_t{t}_num{index}.nii.gz",
        )
        resized = resize_network(network, dmin, dmax, tVol=tissue_volume)

        try:
            with_bolus, max_t0 = bolus_injection(
                resized,
                t,
                alpha=params["alpha"],
                beta=params["beta"],
                V_base=params["v_base"],
                P_flow=params["p_flow"],
                interp_coords_factor=0,
            )
        except (ValueError, IndexError, RuntimeError) as exc:
            print(f"[{index}] skipping: bolus simulation failed ({exc})")
            return

        print(f"[{index}] max contrast arrival time: {max_t0}")
        image = process_network_fluid(with_bolus, dmin, dmax, tVol=tissue_volume, rs=False)
        image = image.astype("float32")

        # Vessel-free voxels are air; vessel voxels carry the contrast concentration.
        volume = np.where(image == 0, AIR_HU, image * CONTRAST_SCALE)
        save_numpy_to_nii(volume, volume_name, header, affine)

        render_projection(volume_name, paths["projections"], config["projection"], device=device)
        print(f"[{index}] t={t} done in {time.time() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic vascular volumes and their DRR projections."
    )
    parser.add_argument("--config", default="config/settings.yaml", help="Engine settings (YAML).")
    parser.add_argument("--reference-volume", required=True,
                        help="NIfTI volume whose header/affine and dimensions define the synthesis grid.")
    parser.add_argument("--output-dir", default="output",
                        help="Root output directory; volumes/ and projections/ are created inside.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of vessel trees to generate (default: value from the config).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    reference = nib.load(args.reference_volume)
    header, affine = reference.header, reference.affine
    tissue_volume = header["dim"][1:4]
    print(f"Synthesis grid: {tuple(tissue_volume)}")

    paths = {
        "volumes": os.path.join(args.output_dir, "volumes"),
        "projections": os.path.join(args.output_dir, "projections"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    num_samples = args.num_samples or config["vessel_generation"]["num_samples"]
    for index in range(num_samples):
        generate_sample(index, config, tissue_volume, header, affine, paths, device=args.device)


if __name__ == "__main__":
    main()
