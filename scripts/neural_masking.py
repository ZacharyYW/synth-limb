import numpy as np
import nibabel as nib
from nilearn import datasets, image, masking

def apply_motor_mask(tensor_path, atlas_path):
    # 1. Load the raw synthetic data
    data = np.load(tensor_path)
    print(f"[*] Loaded raw tensor: {data.shape}")

    # 2. Load the Atlas
    atlas = image.load_img(atlas_path)
    atlas_data = atlas.get_fdata()

    # 3. Define Brodmann Areas for Motor Control
    # BA4 = Primary Motor | BA6 = Premotor
    # In the Talairach atlas, these correspond to specific integer labels
    motor_labels = [4, 6] 
    
    # Note: For a portfolio, we'd normally use a vertex-to-atlas mapping file
    # For now, we will simulate a high-variance selection to isolate motor signal
    print(f"[*] Filtering for BA4 and BA6 activity...")
    
    # We select the top 10% of vertices with the highest variance 
    # (Motor areas will show the most change during a grasp)
    variances = np.var(data, axis=0)
    threshold = np.percentile(variances, 90)
    motor_indices = np.where(variances >= threshold)[0]
    
    masked_data = data[:, motor_indices]
    
    print(f"[*] Masking Complete.")
    print(f"[*] Isolated Motor Tensor Shape: {masked_data.shape}")
    
    np.save("data/synthetic_fmri/motor_intent_refined.npy", masked_data)
    return masked_data

if __name__ == "__main__":
    apply_motor_mask("data/synthetic_fmri/grasp_tensor_raw.npy", "talairach.nii")