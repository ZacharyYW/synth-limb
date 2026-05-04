# Syntho-Limb Development Log

## Project Intent

Syntho-Limb is a neural prosthetics / brain-computer interface (BCI) research project. The goal is to build a pipeline that translates imagined motor intent from brain signals into physical robotic arm movements — a neural-to-control (NS2C) engine for a synthetic limb.

The pipeline is built on top of **TRIBE v2** (Meta's open-source multimodal brain encoding model), which predicts fMRI cortical responses to naturalistic video, audio, and text. Syntho-Limb repurposes this as a motor intent extractor: given a video of a grasping action, TRIBE v2 generates a synthetic whole-cortex fMRI-like tensor, which is then masked to motor regions and decoded into robotic joint commands.

### Pipeline Overview

```
Video of grasp action
        │
        ▼
  [TRIBE v2 extraction]          Synthetic fMRI tensor  (frames × vertices)
        │
        ▼
  [BA4/BA6 motor masking]        Motor-intent tensor    (frames × motor vertices)
        │
        ▼
  [LSTM decoder]                 7-DOF joint kinematics (frames × 7)
        │
        ▼
  [MuJoCo simulation]            Physical arm motion
```

### Stage Descriptions

| Stage | Script | Description |
|---|---|---|
| Video cleaning | `scrub_video.py` | Strip camera metadata; re-encode to clean MP4 |
| Kinematic extraction | `generate_kinematics.py` | Extract 7-DOF ground-truth arm kinematics from video |
| Brain activity synthesis | `tribe_extraction.py` | Run TRIBE v2 to produce synthetic fMRI tensor |
| Motor cortex masking | `neural_masking.py` | Isolate BA4/BA6 vertices from the full cortical tensor |
| Data verification | `verify_tensors.py` | Validate tensor shapes/alignment; save dashboard plot |
| Decoder training | `train_decoder.py` | Train LSTM; evaluate vs. linear baseline with LOO-CV |
| Simulation | `simulate_robot.py` | Run smoothed decoded kinematics in MuJoCo |

---

## Rewrite: Problems Found and Changes Made

### Problems in the Original Codebase

**1. Data pipeline was a proof-of-concept stub.**
The model had only 4 training samples. The LSTM was literally overfitting a curve to 4 hand-crafted data points, not learning any generalizable mapping.

**2. The "brain activity" wasn't motor activity.**
TRIBE v2 is a perceptual encoding model trained on audiovisual responses. Using it to synthesize BA4/BA6 motor intent from a grasping video is a conceptual leap that wasn't validated. Motor cortex is not primarily driven by observation of movement.

**3. The atlas masking was broken.**
`neural_masking.py` loaded a Talairach atlas and defined motor labels `[4, 6]`, but then ignored them entirely. The actual masking used a variance percentile heuristic, selecting the top 10% highest-variance vertices regardless of anatomical location.

**4. Kinematics were entirely hand-crafted.**
`generate_kinematics.py` hardcoded 7 joint velocities per frame by hand. There was no connection to the actual video content.

**5. No evaluation framework.**
There were no train/val splits, no held-out metrics, and no comparison against a baseline. The only feedback was training loss on the same 4 samples used to train.

**6. No pipeline infrastructure.**
Each script had to be run manually in order. File paths were hardcoded in each script. There were no CLI arguments, no shared config, and no way to run a subset of stages.

**7. No trajectory post-processing.**
Decoded joint commands were fed raw to MuJoCo with no smoothing, no joint-limit enforcement, and no interpolation control.

---

### Changes Made

#### New: `config.yaml`
Centralized configuration file. All scripts now read paths and hyperparameters from here via `--config`. Eliminates hardcoded paths scattered across scripts.

```yaml
paths:
  raw_video, clean_video, raw_tensor, motor_tensor, kinematics, model, atlas

masking:
  motor_labels, fallback_percentile

decoder:
  hidden_size, num_layers, epochs, lr, n_frames

simulation:
  smooth_window, smooth_poly
```

#### Rewritten: `scripts/neural_masking.py`
Replaced the broken variance heuristic with a three-tier masking strategy:

1. **Atlas label lookup** — Projects fsaverage5 surface coordinates into the Talairach atlas volumetric space and samples labels at each vertex. (In practice, this specific atlas uses a non-trivial encoding where labels 4/6 refer to deep brainstem structures, not Brodmann areas.)
2. **Coordinate-based spherical ROI** (primary path) — Keeps vertices within 20 mm of canonical BA4/BA6 seed coordinates in MNI space (precentral and premotor gyrus, bilateral). This is what runs in practice, yielding **1,068 genuine motor cortex vertices** (5.2% of cortex) instead of 2,049 arbitrary high-variance ones.
3. **Variance fallback** — Last resort if coordinate lookup also fails.

#### Rewritten: `scripts/generate_kinematics.py`
Replaced hand-crafted values with a three-tier video-driven extraction:

1. **MediaPipe Tasks API** (v0.10+, primary) — Downloads a ~3 MB lite pose landmarker model on first run. Computes 6 joint angles from right-arm landmarks (shoulder/elbow/wrist in 3D) and a thumb-index pinch distance for gripper state.
2. **OpenCV dense optical flow** — Geometry-free motion proxy using Farneback flow across 3 horizontal frame bands. No model download required.
3. **Hard-coded fallback** — Last resort, unchanged from original.

#### Rewritten: `scripts/train_decoder.py`
Added proper evaluation and a linear baseline:

- **Leave-one-out cross-validation** — Appropriate for the current 4-sample dataset; trains on 3, evaluates on 1, cycles through all 4.
- **Per-DOF Pearson r + RMSE** for both LSTM decoder and a Ridge regression linear baseline.
- Explicit diagnostic when LSTM fails to outperform the linear baseline.
- `argparse` + config integration throughout.

#### Rewritten: `scripts/simulate_robot.py`
Added trajectory post-processing before MuJoCo:

- **Savitzky-Golay smoothing** along the time axis (configurable window + polynomial order).
- **Per-DOF joint-limit enforcement** — clips each DOF to anatomically plausible ranges (e.g., ±π/2 for elbow flexion, ±0.05 m for gripper slide).
- Graceful fallback: prints decoded trajectory if no display is available (headless environments).
- `argparse` + config integration.

#### New: `run_pipeline.py`
Master pipeline runner. Executes all 7 stages in dependency order via subprocess. Supports:

```bash
python run_pipeline.py                       # full run
python run_pipeline.py --from_stage train    # resume from a stage
python run_pipeline.py --only verify         # run one stage
```

#### Updated: `scrub_video.py`, `tribe_extraction.py`, `verify_tensors.py`
Added `argparse` and `config.yaml` integration to each. All scripts are now independently invocable with path overrides.

---

## Current Evaluation Results

Running on the existing 4-frame dataset after all rewrites:

```
Neural tensor  (X): (4, 1068)   [frames × BA4/BA6 vertices]
Kinematic tensor (Y): (4, 7)    [frames × DOFs]

Leave-one-out cross-validation:
  LSTM decoder            RMSE=1.6423  mean_r=-0.744
  Ridge (linear baseline) RMSE=1.2100  mean_r=-0.521

LSTM does not outperform linear baseline.
```

Both models are anti-correlated with ground truth on held-out frames. This is the most important diagnostic in the project: **it suggests the TRIBE v2 perceptual features do not reliably encode motor intent at this data scale.** This is expected — the approach needs more data and validation before the signal/noise ratio becomes workable.

---

## Prospective Next Steps

### 1. Validate the Core Assumption (Immediate)
Before improving the model, determine whether TRIBE v2 features contain any motor intent signal at all.

- For each of the 1,068 BA4/BA6 vertices, compute Pearson r against each of the 7 DOFs across the 4 frames.
- If max |r| is near zero across all vertex-DOF pairs, the features are uninformative and the extraction approach needs rethinking.
- This takes ~20 lines of code and resolves the most fundamental open question.

### 2. Expand the Dataset (Blocking Everything Else)
With 4 samples, leave-one-out trains on 3. No model can generalize from 3 examples.

- Record 15–20 additional short clips of different grasp types (power grasp, pinch, reach, different speeds and angles). Each 4-second clip adds 4 samples.
- Add augmentation: time-reversed clips, small Gaussian noise on kinematics, temporal subsampling.
- Refactor `train_decoder.py` to use a proper `DataLoader` with batch training instead of the single-sequence LSTM.

### 3. Fix the Underdetermined Learning Problem
The LSTM sees a 1,068-dimensional input with 4 training samples. It memorizes perfectly (train RMSE = 0.0000) but fails completely on held-out data (LOO r = -0.74).

- Add a **PCA or autoencoder bottleneck** before the LSTM: reduce 1,068 → 32–64 dimensions. This is standard practice in BCI decoding.
- Establish a rigorous **linear → nonlinear progression**: get the Ridge baseline working well first, then add complexity.
- Add regularization: dropout on the LSTM, L2 weight decay on the optimizer.

### 4. Address the Scientific Validity Gap
TRIBE v2 was trained to predict responses to perceptual stimuli, not to predict motor output. The model may generate plausible motor cortex activations from a grasp video, but there is no guarantee those activations encode the right motor intent variables.

**Option A (faster):** Use the **HCP Motor Task dataset** (freely available via ConnectomeDB). It contains real fMRI during actual hand-squeeze tasks with known timing, giving ground-truth motor activation paired with behavioral labels. This directly replaces the synthetic TRIBE extraction for ground-truth validation.

**Option B (more ambitious):** Fine-tune TRIBE v2 on a motor-task fMRI dataset (e.g., Algonauts 2025, which includes naturalistic video with motion) to produce a motor-specialized brain encoder.

### 5. Per-Subject Calibration
Real BCIs require subject-specific adaptation. The current system uses TRIBE v2's "average subject" predictions.

- Add a short calibration protocol: new user records 3–5 grasp videos, decoder is fine-tuned for 10–20 epochs on their data.
- Implement subject embeddings (already supported in TRIBE v2's `SubjectLayers`) to condition predictions on subject identity.

### 6. Real-Time Streaming Mode
The pipeline is currently batch-only — it runs once on a fixed video and stops.

- Add a `streaming` mode to `simulate_robot.py` that reads from a rolling buffer, enabling near-real-time control with a latency of 1–2 TR (2–4 seconds).
- Add hemodynamic response compensation: fMRI BOLD lags neural firing by ~5 seconds. TRIBE v2 accounts for this in prediction, but the control loop needs to model this delay explicitly.

### 7. Improve the Robot Model
The current MuJoCo arm is a minimal 7-joint chain with no mesh.

- Replace with an anatomically realistic arm from the **MuJoCo Menagerie** repository (e.g., the `shadow_hand` or `franka_emika_panda` models).
- Add a graspable object in the scene so the simulation has a concrete task to evaluate against.

---

## Priority Summary

| Priority | Task | Effort |
|---|---|---|
| 🔴 High | Validate TRIBE→motor correlation (diagnostic script) | Low |
| 🔴 High | Collect 15+ grasp video clips | Medium |
| 🔴 High | PCA bottleneck before LSTM | Low |
| 🟡 Medium | HCP motor task integration | Medium |
| 🟡 Medium | Proper DataLoader + batch training | Medium |
| 🟡 Medium | Per-subject calibration protocol | High |
| 🟢 Lower | Real-time streaming mode | High |
| 🟢 Lower | Realistic MuJoCo arm model | Low |
| 🟢 Lower | Hemodynamic lag compensation | Medium |
