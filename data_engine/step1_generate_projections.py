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

The output is one contrast-only projection per rendered contrast phase (early
filling, peak, washout; see ``contrast_dynamics`` in the config), which
``step2_compose_xca.py`` then fuses with real contrast-free backgrounds.

The synthesis grid (voxel count and spacing) is defined in the config
(``grid`` section); no reference CT is needed.  ``--reference-volume`` remains
available to copy the grid geometry (header/affine/dimensions) from an existing
NIfTI volume instead.

Usage::

    python step1_generate_projections.py --output-dir output --num-samples 2000
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


def make_synthetic_grid(config):
    """Build the NIfTI header/affine of the synthesis grid from the config.

    The grid is isotropic by default (``grid.size`` voxels of ``grid.spacing_mm``
    millimetres, centred on the origin), which is all the voxeliser and the DRR
    renderer need; no reference CT is required.
    """
    grid = config.get("grid", {})
    size = np.asarray(grid.get("size", [256, 256, 256]), dtype=int)
    spacing = np.asarray(grid.get("spacing_mm", [1.0, 1.0, 1.0]), dtype=float)
    if size.shape != (3,) or spacing.shape != (3,):
        raise ValueError("grid.size and grid.spacing_mm must have three entries")
    affine = np.diag([*spacing, 1.0])
    if grid.get("centre_origin", True):
        affine[:3, 3] = -(size - 1) * spacing / 2.0
    header = nib.Nifti1Header()
    header.set_data_shape(tuple(int(v) for v in size))
    header.set_zooms(tuple(float(v) for v in spacing))
    header.set_xyzt_units("mm")
    header.set_sform(affine, code=1)
    header.set_qform(affine, code=1)
    return header, affine, size


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


def render_projection(volume_path, config, device="cuda"):
    """Render ``volume_path`` from every configured viewpoint.

    Returns a list of ``(rotation, projection)`` with the raw (un-normalised) 2-D
    projections, or an empty list if the volume holds no contrast.  Randomised
    rotations emulate the varying fluoroscopic viewpoints and patient positioning
    seen across acquisitions.
    """
    sdd = config["sdd"]
    translate = [[0.0, sdd - config["translation"], 0.0]]
    out = []
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
        projection = projection.float().cpu()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        span = float(projection.max() - projection.min())
        if not (span > 0):
            # Contrast-free volume (before arrival / after washout): a constant
            # projection would become NaN -> all-black PNG.  Skip it instead.
            print(f"  skipping empty projection of {os.path.basename(volume_path)}")
            continue
        out.append((rotation, projection))
    return out


def save_projections(rendered, output_folder, config, scale=None):
    """Normalise and save rendered projections as PNG.

    ``scale`` is the value mapped to 255.  ``None`` normalises every frame by its
    own maximum (the original behaviour, which flattens the intensity differences
    between the contrast phases of a tree); with ``projection.normalise:
    per_tree`` the caller passes the maximum over all phases of the tree so that
    early-filling and washout frames stay fainter than the peak frame.
    """
    os.makedirs(output_folder, exist_ok=True)
    enhance = config.get("contrast_enhance", 3.0)
    for volume_name, rotation, projection in rendered:
        top = float(projection.max()) if scale is None else float(scale)
        normalized = (255 * (projection - projection.min()) / max(top - float(projection.min()), 1e-6)).clamp(0, 255).byte()
        image = Image.fromarray(normalized.numpy(), "L")
        image = ImageEnhance.Contrast(image).enhance(enhance)
        file_name = os.path.basename(volume_name)[:-7] + str(rotation) + ".png"
        image.save(os.path.join(output_folder, file_name))


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


def sample_timesteps(timing, params, max_t0, rng):
    """Timesteps at which a tree is rendered.

    ``t_mode: absolute`` uses the fixed grid ``t_start .. t_end`` step ``t_step``
    (original behaviour; most of those frames fall after the washout of trees
    whose contrast arrives within ~30-90 frames, so they are near-identical).
    ``t_mode: relative`` (default) draws ``frames_per_tree`` phases ``p`` from
    ``phase_range`` and renders at ``t = p * (max_t0 + alpha * beta)``, i.e. with
    respect to the moment the last-arriving segment peaks: ``p < 1`` gives
    early filling (only the proximal tree opacified), ``p ~ 1`` the peak,
    ``p > 1`` the washout with the trunk already fading.  ``phase_anchor``
    optionally keeps one frame exactly at the peak.
    """
    if timing.get("t_mode", "relative") == "absolute":
        return list(range(timing["t_start"], timing["t_end"], timing["t_step"]))
    peak = max_t0 + params["alpha"] * params["beta"]
    lo, hi = timing.get("phase_range", [0.3, 2.2])
    n = int(timing.get("frames_per_tree", 4))
    phases = list(rng.uniform(lo, hi, size=max(n - 1, 0) if timing.get("phase_anchor", True) else n))
    if timing.get("phase_anchor", True):
        phases.append(1.0)
    return sorted(set(int(round(p * peak)) for p in phases))


def generate_sample(index, config, tissue_volume, header, affine, paths, device="cuda", rng=None):
    """Generate one vessel tree and render its projection at every timestep."""
    rng = np.random.default_rng() if rng is None else rng
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
    resized = resize_network(network, dmin, dmax, tVol=tissue_volume)

    timing = config["contrast_dynamics"]
    dyn = dict(alpha=params["alpha"], beta=params["beta"], V_base=params["v_base"], P_flow=params["p_flow"],
               interp_coords_factor=0, branch_velocity_jitter=timing.get("branch_velocity_jitter", 0.0),
               dispersion_gain=timing.get("dispersion_gain", 0.0), rng=rng)
    try:
        # First call builds the tree and its dynamics (velocity jitter, arrival times);
        # later timesteps reuse the same tree through ``head``.
        _, max_t0, head = bolus_injection(resized, 0, **dyn)
    except (ValueError, IndexError, RuntimeError) as exc:
        print(f"[{index}] skipping: bolus simulation failed ({exc})")
        return
    timesteps = sample_timesteps(timing, params, max_t0, rng)
    print(f"[{index}] max contrast arrival time: {max_t0:.1f}; timesteps {timesteps}")

    rendered = []
    for t in timesteps:
        volume_name = os.path.join(
            paths["volumes"],
            f"Lnet_d{params['d0']:.2f}_iter{params['niter']}_t{t}_num{index}.nii.gz",
        )
        with_bolus, _, _ = bolus_injection(resized, t, head=head, **dyn)
        image = process_network_fluid(with_bolus, dmin, dmax, tVol=tissue_volume, rs=False)
        image = image.astype("float32")
        if not np.any(image > 0):
            # No contrast anywhere at this timestep (before arrival or after the
            # washout tail): nothing to render.
            print(f"[{index}] t={t}: no contrast in the volume, skipping")
            continue

        # Vessel-free voxels are air; vessel voxels carry the contrast concentration.
        volume = np.where(image == 0, AIR_HU, image * CONTRAST_SCALE)
        save_numpy_to_nii(volume, volume_name, header, affine)

        for rotation, projection in render_projection(volume_name, config["projection"], device=device):
            rendered.append((volume_name, rotation, projection))
        print(f"[{index}] t={t} done in {time.time() - start:.1f}s")

    if not rendered:
        return
    proj_cfg = config["projection"]
    scale = max(float(p.max()) for _, _, p in rendered) if proj_cfg.get("normalise", "per_tree") == "per_tree" else None
    save_projections(rendered, paths["projections"], proj_cfg, scale=scale)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic vascular volumes and their DRR projections (no reference CT needed)."
    )
    parser.add_argument("--config", default="config/settings.yaml", help="Engine settings (YAML).")
    parser.add_argument("--reference-volume", default=None,
                        help="Optional NIfTI volume whose header/affine and dimensions replace the config's grid "
                             "(kept for backwards compatibility; the engine does not need a CT).")
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

    if args.reference_volume:
        reference = nib.load(args.reference_volume)
        header, affine = reference.header, reference.affine
        tissue_volume = np.asarray(header["dim"][1:4], dtype=int)
        print(f"Synthesis grid from {args.reference_volume}: {tuple(tissue_volume)} voxels, "
              f"spacing {tuple(round(float(z), 3) for z in header.get_zooms()[:3])} mm")
    else:
        header, affine, tissue_volume = make_synthetic_grid(config)
        print(f"Synthesis grid from config: {tuple(tissue_volume)} voxels, "
              f"spacing {tuple(config.get('grid', {}).get('spacing_mm', [1.0, 1.0, 1.0]))} mm")

    paths = {
        "volumes": os.path.join(args.output_dir, "volumes"),
        "projections": os.path.join(args.output_dir, "projections"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    num_samples = args.num_samples or config["vessel_generation"]["num_samples"]
    for index in range(num_samples):
        generate_sample(index, config, tissue_volume, header, affine, paths, device=args.device, rng=rng)


if __name__ == "__main__":
    main()
