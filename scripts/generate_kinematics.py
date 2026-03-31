"""
Syntho-Limb NS2C Engine: Sprint 2 - Kinematic Ground Truth
==========================================================
Generates a 7-DOF target velocity matrix aligned to the 4-frame 
synthetic fMRI tensor extracted from `grasp_test_clean.mp4`.
"""

import numpy as np

def generate_grasp_kinematics():
    print("[*] Generating 7-DOF Kinematic Ground Truth...")
    
    # Define the 7-DOF states for our 4 time segments.
    # Format: [Base, Shoulder, Elbow, Wrist_Roll, Wrist_Pitch, Wrist_Yaw, Gripper]
    # Values represent normalized velocities (except Gripper, which is state: 0=Open, 1=Closed)
    
    frame_1_rest =      [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
    frame_2_reach =     [ 0.2,  0.6,  0.5,  0.1,  0.0,  0.0,  0.0]  # Arm extends
    frame_3_grasp =     [ 0.0,  0.1,  0.1,  0.0,  0.2, -0.1,  1.0]  # Arm slows, wrist aligns, gripper closes
    frame_4_retract =   [-0.2, -0.5, -0.4,  0.0,  0.0,  0.0,  1.0]  # Arm reverses, gripper holds
    
    # Stack into a (4, 7) tensor
    kinematics_tensor = np.array([
        frame_1_rest,
        frame_2_reach,
        frame_3_grasp,
        frame_4_retract
    ])
    
    save_path = "data/synthetic_fmri/grasp_kinematics.npy"
    np.save(save_path, kinematics_tensor)
    
    print(f"[*] SUCCESS: Kinematic targets saved to {save_path}")
    print(f"[*] Kinematics Shape: {kinematics_tensor.shape} (Frames, DOFs)")
    
    return kinematics_tensor

if __name__ == "__main__":
    generate_grasp_kinematics()