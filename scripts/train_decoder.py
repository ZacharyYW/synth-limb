"""
Syntho-Limb NS2C Engine: Sprint 3 - Decoder Training
=====================================================
Trains a lightweight LSTM to map BA4/BA6 motor-intent fMRI → 7-DOF kinematics.

Evaluation:
  - Leave-one-out cross-validation (appropriate for small datasets)
  - RMSE and per-DOF Pearson r reported for both LSTM and a Ridge linear baseline
  - Final model trained on all available data and saved for simulation

Usage:
  python scripts/train_decoder.py [--config config.yaml] [--tensor ...] [--kinematics ...] [--model_out ...]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model
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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE + per-DOF Pearson correlation."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    corrs = []
    for dof in range(y_true.shape[-1]):
        t = y_true[..., dof].ravel()
        p = y_pred[..., dof].ravel()
        if np.std(t) > 1e-8 and np.std(p) > 1e-8:
            r, _ = pearsonr(t, p)
        else:
            r = 0.0
        corrs.append(float(r))
    return {"rmse": rmse, "pearson_r": corrs, "mean_r": float(np.mean(corrs))}


def _print_metrics(label: str, m: dict) -> None:
    per_dof = "  ".join(f"{r:+.2f}" for r in m["pearson_r"])
    print(f"  {label:<22}  RMSE={m['rmse']:.4f}  mean_r={m['mean_r']:+.3f}  per-DOF=[{per_dof}]")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_lstm(
    X: np.ndarray, Y: np.ndarray,
    hidden_size: int, num_layers: int,
    epochs: int, lr: float,
    device: torch.device,
) -> NS2CDecoder:
    model = NS2CDecoder(X.shape[1], hidden_size, num_layers, Y.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    Yt = torch.tensor(Y, dtype=torch.float32).unsqueeze(0).to(device)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        criterion(model(Xt), Yt).backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# Leave-one-out cross-validation
# ---------------------------------------------------------------------------

def leave_one_out_eval(
    X: np.ndarray, Y: np.ndarray,
    hidden_size: int, num_layers: int,
    epochs: int, lr: float,
    device: torch.device,
) -> dict:
    n = X.shape[0]
    lstm_preds, linear_preds = [], []

    for held_out in range(n):
        train_idx = [i for i in range(n) if i != held_out]
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_te = X[[held_out]]

        # LSTM
        model = _train_lstm(X_tr, Y_tr, hidden_size, num_layers, epochs, lr, device)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_te, dtype=torch.float32).unsqueeze(0).to(device))
        lstm_preds.append(pred.cpu().numpy()[0])

        # Ridge linear baseline
        ridge = Ridge(alpha=1.0).fit(X_tr, Y_tr)
        linear_preds.append(ridge.predict(X_te))

    return {
        "lstm": compute_metrics(Y, np.concatenate(lstm_preds, axis=0)),
        "linear_baseline": compute_metrics(Y, np.concatenate(linear_preds, axis=0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_engine(
    config_path: str = "config.yaml",
    tensor_path: str | None = None,
    kinematics_path: str | None = None,
    model_path: str | None = None,
) -> None:
    cfg = load_config(config_path)
    tensor_path = tensor_path or cfg["paths"]["motor_tensor"]
    kinematics_path = kinematics_path or cfg["paths"]["kinematics"]
    model_path = model_path or cfg["paths"]["model"]
    dcfg = cfg["decoder"]
    hidden_size: int = dcfg["hidden_size"]
    num_layers: int = dcfg["num_layers"]
    epochs: int = dcfg["epochs"]
    lr: float = dcfg["lr"]

    if not Path(tensor_path).exists() or not Path(kinematics_path).exists():
        print("[!] Missing input data. Run tribe_extraction.py and neural_masking.py first.")
        return

    X = np.load(tensor_path).astype(np.float32)
    Y = np.load(kinematics_path).astype(np.float32)
    print(f"[*] X (neural): {X.shape}   Y (kinematics): {Y.shape}")

    if X.shape[0] != Y.shape[0]:
        print(f"[!] Frame mismatch: X={X.shape[0]}, Y={Y.shape[0]}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # --- LOO evaluation ---
    print(f"\n[*] Leave-one-out cross-validation ({X.shape[0]} folds) ...")
    results = leave_one_out_eval(X, Y, hidden_size, num_layers, epochs, lr, device)
    _print_metrics("LSTM decoder", results["lstm"])
    _print_metrics("Ridge (linear baseline)", results["linear_baseline"])

    if results["lstm"]["mean_r"] > results["linear_baseline"]["mean_r"]:
        print("\n[+] LSTM outperforms linear baseline on LOO-CV.")
    else:
        print("\n[!] LSTM does not outperform linear baseline — consider more training data or tuning.")

    # --- Full training ---
    print(f"\n[*] Training on full dataset ({X.shape[0]} frames, {epochs} epochs) ...")
    model = _train_lstm(X, Y, hidden_size, num_layers, epochs, lr, device)
    # Report training-set fit
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy()[0]
    train_metrics = compute_metrics(Y, train_pred)
    _print_metrics("Final model (train set)", train_metrics)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n[*] Saved decoder weights to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NS2C LSTM decoder.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tensor", help="Override motor tensor path")
    parser.add_argument("--kinematics", help="Override kinematics path")
    parser.add_argument("--model_out", help="Override saved model path")
    args = parser.parse_args()
    train_engine(args.config, args.tensor, args.kinematics, args.model_out)
