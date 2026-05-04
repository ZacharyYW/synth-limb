"""
Syntho-Limb NS2C Engine: Sprint 2 - Kinematic Ground Truth
===========================================================
Extracts 7-DOF arm joint kinematics from video frames.

Strategy (in priority order):
  1. MediaPipe PoseLandmarker + HandLandmarker (Tasks API, mediapipe >= 0.10)
     Downloads a ~3 MB lite model on first run and caches it under models/.
  2. OpenCV dense optical flow — geometry-free motion proxy, no download needed.
  3. Hard-coded fallback grasp sequence.

DOF order:
  0: shoulder flexion     (sagittal angle)
  1: shoulder abduction   (lateral angle)
  2: shoulder rotation    (forearm yaw)
  3: elbow flexion
  4: wrist flexion
  5: wrist deviation
  6: gripper              (thumb-index pinch: 0=open, 1=closed)
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_model(url: str, path: Path) -> bool:
    if path.exists():
        return True
    try:
        print(f"[*] Downloading MediaPipe model → {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"[!] Download failed: {e}")
        return False


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Strategy 1: MediaPipe Tasks API (mediapipe >= 0.10)
# ---------------------------------------------------------------------------

def _mediapipe_kinematics(
    video_path: str,
    n_frames: int,
    model_dir: str = "models",
) -> np.ndarray | None:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        print("[!] mediapipe not installed. pip install mediapipe")
        return None

    pose_path = Path(model_dir) / "pose_landmarker_lite.task"
    hand_path = Path(model_dir) / "hand_landmarker.task"

    if not _ensure_model(POSE_MODEL_URL, pose_path):
        return None
    _ensure_model(HAND_MODEL_URL, hand_path)  # optional — graceful if absent

    # Build pose detector
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_opts)

    hand_detector = None
    if hand_path.exists():
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
        )
        hand_detector = mp_vision.HandLandmarker.create_from_options(hand_opts)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Cannot open {video_path}")
        pose_detector.close()
        if hand_detector:
            hand_detector.close()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total - 1, n_frames, dtype=int)
    kinematics = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        dofs = np.zeros(7)

        if ret:
            rgb = frame[:, :, ::-1].copy()
            h, w = frame.shape[:2]
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_res = pose_detector.detect(mp_img)
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks[0]

                def pt(i: int) -> np.ndarray:
                    return np.array([lm[i].x * w, lm[i].y * h, lm[i].z * w])

                # Right-arm: shoulder=12, elbow=14, wrist=16, hip=24
                shoulder_r, shoulder_l = pt(12), pt(11)
                elbow, wrist, hip_r = pt(14), pt(16), pt(24)
                mid_shoulder = (shoulder_r + shoulder_l) / 2
                forearm = wrist - elbow

                dofs[0] = _angle_between(hip_r, shoulder_r, elbow)
                dofs[1] = _angle_between(mid_shoulder, shoulder_r, elbow)
                dofs[2] = float(np.arctan2(forearm[0], forearm[2] + 1e-8))
                dofs[3] = _angle_between(shoulder_r, elbow, wrist)
                dofs[4] = float(np.arctan2(wrist[1] - elbow[1], abs(wrist[2] - elbow[2]) + 1e-8))
                dofs[5] = float(np.arctan2(wrist[0] - elbow[0], abs(wrist[2] - elbow[2]) + 1e-8))

            if hand_detector:
                hand_res = hand_detector.detect(mp_img)
                if hand_res.hand_landmarks:
                    hl = hand_res.hand_landmarks[0]
                    thumb = np.array([hl[4].x * w, hl[4].y * h])
                    index = np.array([hl[8].x * w, hl[8].y * h])
                    pinch = np.linalg.norm(thumb - index)
                    dofs[6] = float(np.clip(1.0 - pinch / (0.2 * w), 0.0, 1.0))

        kinematics.append(dofs)

    cap.release()
    pose_detector.close()
    if hand_detector:
        hand_detector.close()

    result = np.array(kinematics)
    # Normalise angles (DOF 0–5) to [−1, 1]; gripper already in [0, 1]
    for i in range(6):
        col = result[:, i]
        rng = col.max() - col.min()
        if rng > 1e-6:
            result[:, i] = (col - col.min()) / rng * 2.0 - 1.0
    return result


# ---------------------------------------------------------------------------
# Strategy 2: OpenCV dense optical flow (no model download required)
# ---------------------------------------------------------------------------

def _optical_flow_kinematics(video_path: str, n_frames: int) -> np.ndarray | None:
    """
    Estimate per-DOF kinematics from dense optical flow between consecutive frames.
    Frames are divided into three horizontal bands (shoulder / elbow / wrist region).
    Motion magnitude and direction in each band map to DOF values.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample n_frames + 1 frames so we can compute n_frames inter-frame flows
    indices = np.linspace(0, total - 1, n_frames + 1, dtype=int)

    gray_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(gray_frames) < 2:
        return None

    kinematics = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        h, w = flow.shape[:2]
        # Three vertical bands: left=shoulder, middle=elbow, right=wrist
        bands = [flow[:, :w//3], flow[:, w//3:2*w//3], flow[:, 2*w//3:]]
        dofs = np.zeros(7)
        for j, band in enumerate(bands):
            mag = np.sqrt(band[..., 0]**2 + band[..., 1]**2)
            angle = float(np.arctan2(band[..., 1].mean(), band[..., 0].mean() + 1e-8))
            dofs[j * 2] = float(np.tanh(mag.mean() * 0.3))
            if j * 2 + 1 < 6:
                dofs[j * 2 + 1] = float(np.tanh(angle))
        # Gripper proxy: high wrist-region variance → closing motion
        wrist_mag = np.sqrt(bands[2][..., 0]**2 + bands[2][..., 1]**2)
        dofs[6] = float(np.clip(wrist_mag.mean() * 0.5, 0.0, 1.0))
        kinematics.append(dofs)

    result = np.array(kinematics[:n_frames])
    # Normalise DOF 0–5 to [−1, 1]
    for i in range(6):
        col = result[:, i]
        rng = col.max() - col.min()
        if rng > 1e-6:
            result[:, i] = (col - col.min()) / rng * 2.0 - 1.0
    return result


# ---------------------------------------------------------------------------
# Strategy 3: hard-coded fallback
# ---------------------------------------------------------------------------

def _hardcoded_kinematics() -> np.ndarray:
    return np.array([
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],   # rest
        [ 0.5,  0.3,  0.1,  0.8,  0.1,  0.0,  0.0],   # reach
        [ 0.3,  0.2,  0.0,  0.6,  0.3, -0.1,  1.0],   # grasp
        [-0.2, -0.1, -0.1,  0.3,  0.0,  0.0,  1.0],   # retract
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_grasp_kinematics(
    video_path: str,
    output_path: str,
    n_frames: int = 4,
    model_dir: str = "models",
) -> np.ndarray:
    print(f"[*] Extracting 7-DOF kinematics from: {video_path}")

    kinematics = _mediapipe_kinematics(video_path, n_frames, model_dir)
    if kinematics is not None:
        print(f"[*] MediaPipe (Tasks API) succeeded: {kinematics.shape}")
    else:
        print("[!] MediaPipe failed — trying optical flow fallback ...")
        kinematics = _optical_flow_kinematics(video_path, n_frames)
        if kinematics is not None:
            print(f"[*] Optical flow extraction succeeded: {kinematics.shape}")
        else:
            print("[!] Optical flow failed — using hard-coded fallback.")
            kinematics = _hardcoded_kinematics()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, kinematics)
    print(f"[*] Saved kinematics to {output_path}  shape={kinematics.shape}")
    return kinematics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 7-DOF kinematics from video.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", help="Override video path")
    parser.add_argument("--output", help="Override output path")
    parser.add_argument("--n_frames", type=int, help="Number of time frames to extract")
    parser.add_argument("--model_dir", default="models", help="Directory for MediaPipe model files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_grasp_kinematics(
        video_path=args.video or cfg["paths"]["clean_video"],
        output_path=args.output or cfg["paths"]["kinematics"],
        n_frames=args.n_frames or cfg["decoder"]["n_frames"],
        model_dir=args.model_dir,
    )
