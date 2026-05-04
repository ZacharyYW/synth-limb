"""
Syntho-Limb NS2C Engine: Sprint 4 - MuJoCo Simulation
=====================================================
Runs decoded neural commands on a 7-DOF robotic arm with:
  - Savitzky-Golay trajectory smoothing
  - Per-DOF joint-limit enforcement
  - Linear interpolation between decoded frames at simulation frequency
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.signal import savgol_filter


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model (must match training definition)
# ---------------------------------------------------------------------------

class NS2CDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, output_size: int = 7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out)


# ---------------------------------------------------------------------------
# Robot definition
# ---------------------------------------------------------------------------

MJCF_MODEL = """
<mujoco>
    <option gravity="0 0 -9.81" />
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
        <body name="base" pos="0 0 0.1">
            <geom type="cylinder" size="0.1 0.1" rgba="0 0 1 1"/>
            <joint name="joint1" type="hinge" axis="0 0 1"/>
            <body name="shoulder" pos="0 0 0.2">
                <geom type="capsule" size="0.05 0.2" fromto="0 0 0 0 0 0.4" rgba="1 0 0 1"/>
                <joint name="joint2" type="hinge" axis="0 1 0"/>
                <body name="elbow" pos="0 0 0.4">
                    <geom type="capsule" size="0.04 0.15" fromto="0 0 0 0 0 0.3" rgba="0 1 0 1"/>
                    <joint name="joint3" type="hinge" axis="0 1 0"/>
                    <body name="wrist" pos="0 0 0.3">
                        <geom type="sphere" size="0.05" rgba="1 1 0 1"/>
                        <joint name="joint4" type="hinge" axis="1 0 0"/>
                        <joint name="joint5" type="hinge" axis="0 1 0"/>
                        <joint name="joint6" type="hinge" axis="0 0 1"/>
                        <body name="gripper" pos="0 0 0.05">
                            <geom name="hand" type="box" size="0.02 0.05 0.05" rgba="1 0 1 1"/>
                            <joint name="joint7" type="slide" axis="0 1 0"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <position name="a1" joint="joint1" kp="100"/>
        <position name="a2" joint="joint2" kp="100"/>
        <position name="a3" joint="joint3" kp="100"/>
        <position name="a4" joint="joint4" kp="50"/>
        <position name="a5" joint="joint5" kp="50"/>
        <position name="a6" joint="joint6" kp="50"/>
        <position name="a7" joint="joint7" kp="50"/>
    </actuator>
</mujoco>
"""

# (lower, upper) bounds per DOF — radians for revolute, metres for slide
JOINT_LIMITS = np.array([
    [-np.pi,    np.pi   ],   # base rotation
    [-np.pi/2,  np.pi/2 ],   # shoulder flexion
    [-np.pi/2,  np.pi/2 ],   # elbow flexion
    [-np.pi/2,  np.pi/2 ],   # wrist roll
    [-np.pi/4,  np.pi/4 ],   # wrist pitch
    [-np.pi/4,  np.pi/4 ],   # wrist yaw
    [-0.05,     0.05    ],   # gripper slide
])

DOF_LABELS = ["Base", "Shoulder", "Elbow", "WristRoll", "WristPitch", "WristYaw", "Gripper"]


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def smooth_trajectory(predictions: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along the time axis."""
    n = predictions.shape[0]
    # window must be odd and ≥ poly+1; clamp to available frames
    window = min(window if window % 2 == 1 else window + 1, n if n % 2 == 1 else n - 1)
    if window < poly + 2:
        return predictions
    return savgol_filter(predictions, window_length=window, polyorder=poly, axis=0)


def enforce_joint_limits(predictions: np.ndarray) -> np.ndarray:
    for dof in range(predictions.shape[1]):
        lo, hi = JOINT_LIMITS[dof]
        predictions[:, dof] = np.clip(predictions[:, dof], lo, hi)
    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_neuro_simulation(
    config_path: str = "config.yaml",
    tensor_path: str | None = None,
    model_path: str | None = None,
) -> None:
    cfg = load_config(config_path)
    tensor_path = tensor_path or cfg["paths"]["motor_tensor"]
    model_path = model_path or cfg["paths"]["model"]
    scfg = cfg.get("simulation", {})
    smooth_window: int = scfg.get("smooth_window", 5)
    smooth_poly: int = scfg.get("smooth_poly", 2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_np = np.load(tensor_path)

    if not Path(model_path).exists():
        print(f"[!] Model not found at {model_path}. Run train_decoder.py first.")
        return

    model = NS2CDecoder(input_size=X_np.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0).to(device)
        raw_preds = model(X_t).cpu().numpy()[0]  # (T, 7)

    predictions = enforce_joint_limits(smooth_trajectory(raw_preds, smooth_window, smooth_poly))
    n_frames = predictions.shape[0]
    print(f"[*] Decoded {n_frames} frames (smoothed + joint-limited)")
    for t, row in enumerate(predictions):
        print(f"    t={t}: " + "  ".join(f"{l}={v:+.3f}" for l, v in zip(DOF_LABELS, row)))

    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("\n[!] mujoco not installed — trajectory printed above. pip install mujoco")
        return

    mj_model = mujoco.MjModel.from_xml_string(MJCF_MODEL)
    mj_data = mujoco.MjData(mj_model)

    print("\n[*] Launching MuJoCo viewer ...")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        start_time = time.time()
        duration = float(n_frames)

        while viewer.is_running() and (time.time() - start_time) < duration:
            step_start = time.time()
            elapsed = mj_data.time

            frame_idx = min(int(elapsed), n_frames - 2)
            alpha = elapsed - frame_idx
            targets = (1 - alpha) * predictions[frame_idx] + alpha * predictions[frame_idx + 1]

            mj_data.ctrl[:] = targets
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            wait = mj_model.opt.timestep - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)

    print("[*] Simulation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuJoCo simulation from decoded kinematics.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tensor", help="Override motor tensor path")
    parser.add_argument("--model", help="Override trained model path")
    args = parser.parse_args()
    run_neuro_simulation(args.config, args.tensor, args.model)
