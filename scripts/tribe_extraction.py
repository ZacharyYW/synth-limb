import os
import numpy as np
import torch
import nibabel as nib
from nilearn import image, masking
from tribev2 import TribeModel

class DigitalDissectionPipeline:
    def __init__(self, model_id: str = "facebook/tribev2", cache_dir: str = "../models"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[*] Initializing TRIBE v2 pipeline on {self.device.upper()}...")
        
        # Load the TRIBE v2 foundation model
        self.model = TribeModel.from_pretrained(model_id, cache_folder=cache_dir)
        
        # Load the Talairach atlas directly from the local file we downloaded
        print("[*] Loading local Talairach neuroanatomical atlas...")
        atlas_path = "talairach.nii" # Points to the file in your syntho-limb directory
        
        if not os.path.exists(atlas_path):
             raise FileNotFoundError(f"Atlas file not found at {atlas_path}. Please ensure you ran the curl command.")
             
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
    
    # Path to your test video
    video_path = "../data/raw_video/grasp_test.mp4"
    
    # Check if the video exists and run the extraction
    if os.path.exists(video_path):
        # 1. Generate the whole-brain synthetic response
        raw_tensor = pipeline.extract_synthetic_brain_activity(video_path)
        
        # 2. Save the tensor to disk for Sprint 2
        save_path = "../data/synthetic_fmri/grasp_tensor_raw.npy"
        np.save(save_path, raw_tensor)
        
        print(f"[*] SUCCESS: Synthetic brain data saved to {save_path}")
        print(f"[*] Tensor Shape: {raw_tensor.shape} (Frames, Voxels)")
    else:
        print(f"[!] Waiting for video. Please place a file at: {video_path}")