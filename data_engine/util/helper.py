"""Small I/O helpers for the XCA data engine."""

import os

import nibabel as nib
import numpy as np


def save_numpy_to_nii(data, output_path, header=None, affine=None):
    """Save a 3-D NumPy array as a ``.nii.gz`` volume.

    Args:
        data: 3-D array to save.
        output_path: Destination path.
        header: Optional NIfTI header to copy (e.g. from the reference volume).
        affine: Optional 4x4 affine; the identity matrix is used when omitted.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D NumPy array.")

    if affine is None:
        affine = np.eye(4)

    nib.save(nib.Nifti1Image(data, affine, header), output_path)
    print(f"NIfTI file saved at: {output_path}")


def find_files_by_prefix(name, imgpath, case_sensitive=True):
    """List the files in ``imgpath`` whose name starts with ``name``.

    Args:
        name: File-name prefix to match (e.g. ``"Lnet"``).
        imgpath: Directory to search.
        case_sensitive: Whether the prefix match is case sensitive.

    Returns:
        Full paths of the matching files, sorted by file name.
    """
    matches = []
    target = name if case_sensitive else name.lower()

    for entry in os.scandir(os.path.normpath(imgpath)):
        if not entry.is_file():
            continue
        compare_name = entry.name if case_sensitive else entry.name.lower()
        if compare_name.startswith(target):
            matches.append(entry.path)

    return sorted(matches, key=os.path.basename)
