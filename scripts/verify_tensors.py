import numpy as np
import matplotlib.pyplot as plt
import os

def verify_data_pipeline():
    neural_path = "data/synthetic_fmri/motor_intent_refined.npy"
    kinematic_path = "data/synthetic_fmri/grasp_kinematics.npy"

    if not os.path.exists(neural_path) or not os.path.exists(kinematic_path):
        print("[!] Error: Missing .npy files. Ensure Sprint 1 & 2 extractions are complete.")
        return

    # Load Tensors
    X_neural = np.load(neural_path)
    Y_kinematics = np.load(kinematic_path)

    print("-" * 40)
    print("DATA VERIFICATION REPORT")
    print("-" * 40)
    print(f"[*] Neural Tensor Shape (X):     {X_neural.shape}")
    print(f"[*] Kinematic Tensor Shape (Y):  {Y_kinematics.shape}")
    
    # Validation Check
    if X_neural.shape[0] != Y_kinematics.shape[0]:
        print("[!] FATAL ERROR: Time-step mismatch. X and Y frames do not align.")
        return
    else:
        print("[*] PASS: Time-steps perfectly aligned.")

    # --- Visualization Dashboard ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    time_steps = np.arange(1, X_neural.shape[0] + 1)

    # Plot 1: Kinematics
    dof_labels = ['Base', 'Shoulder', 'Elbow', 'Wrist Roll', 'Wrist Pitch', 'Wrist Yaw', 'Gripper (1=Closed)']
    for i in range(7):
        ax1.plot(time_steps, Y_kinematics[:, i], label=dof_labels[i], marker='o', linewidth=2)
    ax1.set_title("Target Kinematics (7-DOF)")
    ax1.set_ylabel("Velocity / State")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize='small')

    # Plot 2: Neural Activity (Plotting top 5 most active vertices to avoid visual clutter)
    variances = np.var(X_neural, axis=0)
    top_5_idx = np.argsort(variances)[-5:]
    
    for idx in top_5_idx:
        ax2.plot(time_steps, X_neural[:, idx], marker='x', linestyle='--')
    ax2.set_title("Synthetic Neural Activity (Top 5 BA4/BA6 Vertices)")
    ax2.set_xlabel("Time Segment (Frames)")
    ax2.set_ylabel("Simulated BOLD Response")
    ax2.grid(True, alpha=0.3)

    plt.xticks(time_steps)
    plt.tight_layout()
    
    # Save the plot so you can view it easily
    save_img = "data/verification_plot.png"
    plt.savefig(save_img)
    print(f"[*] SUCCESS: Dashboard rendered and saved to {save_img}")
    plt.show()

if __name__ == "__main__":
    verify_data_pipeline()