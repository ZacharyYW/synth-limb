import os
import numpy as np
import torch
from nilearn import image
from tribev2 import TribeModel


class DigitalDissectionPipeline:
    def __init__(
        self,
        model_id: str = "facebook/tribev2",
        cache_dir: str = "../models",
    ):
        # Brain model can try MPS on Apple Silicon
        self.brain_device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Safest choice today: keep extractor backbones off CUDA.
        # On Mac, CPU is the most reliable until upstream MPS support lands cleanly.
        self.extractor_device = "cpu"

        print(f"[*] Initializing TRIBE v2 pipeline...")
        print(f"    brain_device     = {self.brain_device}")
        print(f"    extractor_device = {self.extractor_device}")

        self.model = TribeModel.from_pretrained(
            model_id,
            cache_folder=cache_dir,
            device=self.brain_device,
            config_update={
                # text backbone
                "data.text_feature.device": self.extractor_device,
                # audio backbone
                "data.audio_feature.device": self.extractor_device,
                # image backbone used by image_feature
                "data.image_feature.image.device": self.extractor_device,
                # image/video backbone used by video_feature
                "data.video_feature.image.device": self.extractor_device,
            },
        )

        print("[*] Loading local Talairach neuroanatomical atlas...")
        atlas_path = "talairach.nii"
        if not os.path.exists(atlas_path):
            raise FileNotFoundError(
                f"Atlas file not found at {atlas_path}. "
                "Please ensure the local file exists."
            )

        self.atlas_img = image.load_img(atlas_path)
        print("[*] Atlas loaded successfully.")

    def extract_synthetic_brain_activity(self, video_path: str) -> np.ndarray:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"[*] Extracting features from: {video_path}")
        events_df = self.model.get_events_dataframe(video_path=video_path)
        preds, segments = self.model.predict(events=events_df)
        print(f"[*] Generated synthetic voxel tensor: {preds.shape}")
        return preds


if __name__ == "__main__":
    pipeline = DigitalDissectionPipeline()
    print("[*] System Ready. Awaiting motor-action video input.")

    video_path = "data/raw_video/grasp_test_clean.mp4"

    if os.path.exists(video_path):
        raw_tensor = pipeline.extract_synthetic_brain_activity(video_path)

        save_path = "data/synthetic_fmri/grasp_tensor_raw.npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, raw_tensor)

        print(f"[*] SUCCESS: Synthetic brain data saved to {save_path}")
        print(f"[*] Tensor Shape: {raw_tensor.shape} (Frames, Voxels)")
    else:
        print(f"[!] Waiting for video. Please place a file at: {os.path.abspath(video_path)}")