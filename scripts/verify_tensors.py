"""
Syntho-Limb NS2C Engine: Data Verification Dashboard
=====================================================
Validates shape/alignment of neural and kinematic tensors and
renders a two-panel plot saved to data/verification_plot.png.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def verify_data_pipeline(
    neural_path: str,
    kinematic_path: str,
    plot_path: str = "data/verification_plot.png",
) -> bool:
    if not os.path.exists(neural_path) or not os.path.exists(kinematic_path):
        print("[!] Missing .npy files — run tribe_extraction and neural_masking first.")
        return False

    X = np.load(neural_path)
    Y = np.load(kinematic_path)

    print("-" * 50)
    print("DATA VERIFICATION REPORT")
    print("-" * 50)
    print(f"  Neural tensor  (X): {X.shape}   dtype={X.dtype}")
    print(f"  Kinematic tensor (Y): {Y.shape}   dtype={Y.dtype}")

    if X.shape[0] != Y.shape[0]:
        print(f"[!] FAIL: frame mismatch — X has {X.shape[0]}, Y has {Y.shape[0]}")
        return False
    print("[+] PASS: frame counts match.")

    if np.isnan(X).any():
        print("[!] WARN: NaNs detected in neural tensor.")
    if np.isnan(Y).any():
        print("[!] WARN: NaNs detected in kinematic tensor.")

    # --- Visualisation ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    t = np.arange(1, X.shape[0] + 1)

    dof_labels = ["Base", "Shoulder", "Elbow", "WristRoll", "WristPitch", "WristYaw", "Gripper"]
    for i in range(Y.shape[1]):
        ax1.plot(t, Y[:, i], marker="o", linewidth=2, label=dof_labels[i])
    ax1.set_title("Target Kinematics (7-DOF)")
    ax1.set_ylabel("Normalised value")
    ax1.legend(fontsize="small", loc="upper right")
    ax1.grid(alpha=0.3)

    top5 = np.argsort(np.var(X, axis=0))[-5:]
    for idx in top5:
        ax2.plot(t, X[:, idx], marker="x", linestyle="--", label=f"vertex {idx}")
    ax2.set_title("Synthetic Neural Activity — Top-5 BA4/BA6 Vertices by Variance")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Simulated BOLD response")
    ax2.legend(fontsize="small", loc="upper right")
    ax2.grid(alpha=0.3)

    plt.xticks(t)
    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=120)
    print(f"[*] Verification plot saved to {plot_path}")
    plt.show()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify pipeline tensor shapes and alignment.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--neural", help="Override neural tensor path")
    parser.add_argument("--kinematics", help="Override kinematics path")
    parser.add_argument("--plot", default="data/verification_plot.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    verify_data_pipeline(
        neural_path=args.neural or cfg["paths"]["motor_tensor"],
        kinematic_path=args.kinematics or cfg["paths"]["kinematics"],
        plot_path=args.plot,
    )
