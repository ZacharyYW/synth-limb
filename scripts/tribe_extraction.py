"""
Syntho-Limb NS2C Engine: Sprint 1 - Synthetic fMRI Extraction
==============================================================
Uses TRIBE v2 (facebook/tribev2) to generate a synthetic whole-cortex
fMRI-like tensor from a video of motor actions.

Output: (n_frames, n_vertices) NumPy array saved to data/synthetic_fmri/.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from nilearn import image


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class DigitalDissectionPipeline:
    def __init__(self, model_id: str = "facebook/tribev2", cache_dir: str = "models") -> None:
        self.brain_device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Keep feature extractors on CPU for broad Mac compatibility
        self.extractor_device = "cpu"

        print(f"[*] Initializing TRIBE v2 pipeline ...")
        print(f"    brain_device     = {self.brain_device}")
        print(f"    extractor_device = {self.extractor_device}")

        from tribev2 import TribeModel

        self.model = TribeModel.from_pretrained(
            model_id,
            cache_folder=cache_dir,
            device=self.brain_device,
            config_update={
                "data.text_feature.device": self.extractor_device,
                "data.audio_feature.device": self.extractor_device,
                "data.image_feature.image.device": self.extractor_device,
                "data.video_feature.image.device": self.extractor_device,
            },
        )

    def extract(self, video_path: str) -> np.ndarray:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        print(f"[*] Extracting brain activity from: {video_path}")
        events_df = self.model.get_events_dataframe(video_path=video_path)
        preds, _ = self.model.predict(events=events_df)
        print(f"[*] Synthetic tensor shape: {preds.shape}  (frames × vertices)")
        return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract synthetic fMRI tensor with TRIBE v2.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", help="Override clean video path")
    parser.add_argument("--output", help="Override raw tensor output path")
    parser.add_argument("--model_id", default="facebook/tribev2", help="HuggingFace model ID")
    args = parser.parse_args()

    cfg = load_config(args.config)
    video_path = args.video or cfg["paths"]["clean_video"]
    output_path = args.output or cfg["paths"]["raw_tensor"]

    pipeline = DigitalDissectionPipeline(model_id=args.model_id)

    if os.path.exists(video_path):
        tensor = pipeline.extract(video_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, tensor)
        print(f"[*] Saved to {output_path}")
    else:
        print(f"[!] Video not found: {os.path.abspath(video_path)}")
        print(f"    Run scripts/scrub_video.py first, or place a video at that path.")
