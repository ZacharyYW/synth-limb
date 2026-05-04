"""
Syntho-Limb NS2C Engine: Sprint 1 - Motor Cortex Masking
=========================================================
Extracts motor-intent signal (BA4/BA6) from a synthetic fMRI tensor by
projecting fsaverage5 surface coordinates into the Talairach atlas.

Falls back to a variance-threshold heuristic if atlas masking yields 0 hits
(e.g. if the atlas coordinate system doesn't overlap the fsaverage surface).
"""

import argparse
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _mni_coords_for_fsaverage5(n_vertices: int) -> np.ndarray | None:
    """Return (n_vertices, 3) MNI mm coordinates for fsaverage5 surface vertices."""
    try:
        from nilearn import datasets
        from nilearn.surface import load_surf_mesh
    except ImportError:
        return None
    fsav = datasets.fetch_surf_fsaverage("fsaverage5")
    lh, _ = load_surf_mesh(fsav.pial_left)
    rh, _ = load_surf_mesh(fsav.pial_right)
    return np.concatenate([lh, rh], axis=0)[:n_vertices].astype(np.float64)


def _atlas_motor_indices(
    atlas_path: str, motor_labels: list, n_vertices: int
) -> tuple[np.ndarray | None, str | None]:
    """
    Map fsaverage5 surface vertices to Talairach atlas labels and return
    indices of vertices in the requested regions.

    Returns (indices, None) on success or (None, error_message) on failure.
    """
    coords = _mni_coords_for_fsaverage5(n_vertices)
    if coords is None:
        return None, "nilearn not available"

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    inv_affine = np.linalg.inv(atlas_img.affine.astype(np.float64))

    ones = np.ones((len(coords), 1), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vox = np.round((inv_affine @ np.hstack([coords, ones]).T).T[:, :3]).astype(int)
    vox = np.clip(vox, 0, np.array(atlas_data.shape[:3]) - 1)

    labels = atlas_data[vox[:, 0], vox[:, 1], vox[:, 2]]
    indices = np.where(np.isin(labels, motor_labels))[0]
    return indices, None


# MNI coordinates of BA4 / BA6 centres (bilateral, precentral + premotor gyrus)
_MOTOR_CORTEX_SEEDS = np.array([
    [ 40, -14,  62],  # right BA4
    [-40, -14,  62],  # left  BA4
    [ 30,  -4,  60],  # right BA6
    [-30,  -4,  60],  # left  BA6
], dtype=np.float64)
_MOTOR_ROI_RADIUS_MM = 20.0


def _roi_motor_indices(n_vertices: int) -> np.ndarray | None:
    """
    Spherical ROI approach: keep fsaverage5 vertices within 20 mm of any
    canonical BA4/BA6 seed coordinate in MNI space.
    Used when atlas label lookup returns 0 vertices.
    """
    coords = _mni_coords_for_fsaverage5(n_vertices)
    if coords is None:
        return None
    # distance from each vertex to its nearest seed
    dists = np.sqrt(
        ((coords[:, None, :] - _MOTOR_CORTEX_SEEDS[None, :, :]) ** 2).sum(axis=-1)
    ).min(axis=1)
    return np.where(dists <= _MOTOR_ROI_RADIUS_MM)[0]


def apply_motor_mask(
    tensor_path: str,
    atlas_path: str,
    motor_labels: list,
    fallback_percentile: int = 90,
    output_path: str = "data/synthetic_fmri/motor_intent_refined.npy",
) -> np.ndarray:
    data = np.load(tensor_path)
    n_vertices = data.shape[1]
    print(f"[*] Loaded raw tensor: {data.shape}")

    motor_indices, err = _atlas_motor_indices(atlas_path, motor_labels, n_vertices)

    if err or motor_indices is None or len(motor_indices) == 0:
        reason = err if err else "0 vertices matched atlas labels"
        print(f"[!] Atlas label lookup failed ({reason}). Trying coordinate-based ROI ...")
        motor_indices = _roi_motor_indices(n_vertices)

    if motor_indices is not None and len(motor_indices) > 0:
        pct = 100 * len(motor_indices) / n_vertices
        print(
            f"[*] Coordinate ROI (BA4/BA6 spheres): {len(motor_indices)} vertices "
            f"({pct:.1f}% of cortex)."
        )
    else:
        print("[!] Coordinate ROI also failed. Falling back to variance threshold.")
        variances = np.var(data, axis=0)
        threshold = np.percentile(variances, fallback_percentile)
        motor_indices = np.where(variances >= threshold)[0]
        print(
            f"[*] Variance fallback: kept {len(motor_indices)} vertices "
            f"(top {100 - fallback_percentile}% by variance)."
        )


    masked = data[:, motor_indices]
    print(f"[*] Motor tensor shape: {masked.shape}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, masked)
    print(f"[*] Saved to {output_path}")
    return masked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply BA4/BA6 motor cortex mask.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tensor", help="Override input tensor path")
    parser.add_argument("--atlas", help="Override atlas path")
    parser.add_argument("--output", help="Override output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_motor_mask(
        tensor_path=args.tensor or cfg["paths"]["raw_tensor"],
        atlas_path=args.atlas or cfg["paths"]["atlas"],
        motor_labels=cfg["masking"]["motor_labels"],
        fallback_percentile=cfg["masking"]["fallback_percentile"],
        output_path=args.output or cfg["paths"]["motor_tensor"],
    )
