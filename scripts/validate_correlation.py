"""
Syntho-Limb NS2C Engine: Phase 1 - Pearson Correlation Diagnostic
==================================================================
Validates whether TRIBE v2 motor-cortex features encode motor intent.

For each of the 1,068 BA4/BA6 vertices, computes Pearson r against each
of the 7 kinematic DOFs across all available frames.  Outputs a formatted
text report to stdout and saves the full (n_vertices, 7) correlation matrix
for downstream inspection.

Benchmark thresholds
--------------------
  PASS  (signal exists)  : ≥3 DOFs with max |r| > 0.30
  WARN  (weak signal)    : 1–2 DOFs with max |r| > 0.30
  FAIL  (no signal)      : all DOFs with max |r| ≤ 0.15
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

DOF_LABELS = ["Base", "Shoulder", "Elbow", "WristRoll", "WristPitch", "WristYaw", "Gripper"]

PASS_THRESHOLD = 0.30
FAIL_THRESHOLD = 0.15
PASS_DOF_COUNT = 3


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between two 1-D arrays; returns 0.0 if either is constant."""
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_correlation_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return (n_vertices, n_dofs) matrix of Pearson r values."""
    n_verts = X.shape[1]
    n_dofs = Y.shape[1]
    R = np.zeros((n_verts, n_dofs), dtype=np.float32)
    for v in range(n_verts):
        for d in range(n_dofs):
            R[v, d] = pearson_r(X[:, v], Y[:, d])
    return R


def classify_signal(n_strong_dofs: int) -> str:
    if n_strong_dofs >= PASS_DOF_COUNT:
        return "PASS"
    elif n_strong_dofs >= 1:
        return "WARN"
    return "FAIL"


def run_validation(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    tensor_path = cfg["paths"]["motor_tensor"]
    kin_path = cfg["paths"]["kinematics"]
    out_path = Path(tensor_path).parent / "correlation_map.npy"

    print("[*] Loading tensors ...")
    X = np.load(tensor_path).astype(np.float64)
    Y = np.load(kin_path).astype(np.float64)
    print(f"    Motor tensor  X: {X.shape}  dtype={X.dtype}")
    print(f"    Kinematics    Y: {Y.shape}  dtype={Y.dtype}")

    if X.shape[0] != Y.shape[0]:
        print(f"[!] Frame count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}. Aborting.")
        return

    n_frames, n_verts = X.shape
    n_dofs = Y.shape[1]
    print(f"[*] Computing Pearson r for {n_verts} vertices × {n_dofs} DOFs ({n_frames} frames) ...")

    R = compute_correlation_matrix(X, Y)
    np.save(out_path, R)
    print(f"[+] Correlation map saved → {out_path}")

    abs_R = np.abs(R)
    print("\n" + "=" * 62)
    print("  PER-DOF SUMMARY")
    print("=" * 62)
    print(f"  {'DOF':<12}  {'max|r|':>8}  {'mean|r|':>8}  {'%|r|>0.30':>10}  {'status':>6}")
    print("  " + "-" * 58)
    strong_dofs = 0
    for d in range(n_dofs):
        col = abs_R[:, d]
        max_r = col.max()
        mean_r = col.mean()
        pct = 100.0 * (col > PASS_THRESHOLD).mean()
        status = "STRONG" if max_r > PASS_THRESHOLD else ("WEAK" if max_r > FAIL_THRESHOLD else "NONE")
        if max_r > PASS_THRESHOLD:
            strong_dofs += 1
        print(f"  {DOF_LABELS[d]:<12}  {max_r:>8.4f}  {mean_r:>8.4f}  {pct:>9.1f}%  {status:>6}")

    print("\n" + "=" * 62)
    print("  TOP 10 VERTEX-DOF PAIRS BY |r|")
    print("=" * 62)
    flat_idx = np.argsort(abs_R.ravel())[::-1][:10]
    for rank, fi in enumerate(flat_idx, 1):
        v, d = divmod(fi, n_dofs)
        print(f"  {rank:2d}. vertex={v:5d}  DOF={DOF_LABELS[d]:<12}  r={R[v, d]:+.4f}")

    verdict = classify_signal(strong_dofs)
    print("\n" + "=" * 62)
    print(f"  VERDICT: {verdict}  ({strong_dofs}/{n_dofs} DOFs with max|r| > {PASS_THRESHOLD})")
    if verdict == "PASS":
        print("  [+] TRIBE features contain detectable motor-intent signal.")
        print("      Proceed: expand data (Phase 2) + PCA (Phase 3).")
    elif verdict == "WARN":
        print("  [!] Weak signal detected in a minority of DOFs.")
        print("      Recommend: expand data before committing to synthetic path.")
    else:
        print("  [!] No motor-intent signal found in TRIBE features.")
        print("      Recommend: pivot to HCP real fMRI dataset (Phase 4).")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pearson correlation diagnostic for TRIBE motor features.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_validation(args.config)
