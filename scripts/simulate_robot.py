"""
Syntho-Limb NS2C Engine: Sprint 3 - MuJoCo Simulation
=====================================================
Executes decoded neural commands on a 7-DOF robotic arm 
using the industry-standard MuJoCo physics engine.
"""

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import time

# 1. Define the Decoder (same as training)
class NS2CDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=7):
        super(NS2CDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out)

# 2. Minimalist 7-DOF Robotic Arm Model (MJCF XML)
# This creates a simple chain of 7 joints in space
mjcf_model = """
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

def run_neuro_simulation():
    # 3. Load Trained Model and Predict
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_np = np.load("data/synthetic_fmri/motor_intent_refined.npy")
    
    model = NS2CDecoder(input_size=X_np.shape[1]).to(device)
    model.load_state_dict(torch.load("models/ns2c_decoder_v1.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0).to(device)
        predictions = model(X_tensor).cpu().numpy()[0] # (4, 7)

    # 4. Initialize MuJoCo
    model = mujoco.MjModel.from_xml_string(mjcf_model)
    data = mujoco.MjData(model)

    print("[*] Launching MuJoCo Viewer. Watch the arm execute your intent...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        # We simulate 4 seconds total (1 second per neural frame)
        duration = 4.0 
        
        while viewer.is_running() and (time.time() - start_time) < duration:
            step_start = time.time()
            elapsed = step_start - start_time
            
            # Interpolate between the 4 frames to get a smooth trajectory
            frame_idx = min(int(elapsed), 3)
            next_frame_idx = min(frame_idx + 1, 3)
            alpha = elapsed - frame_idx # Decimal progress between frames
            
            # Linear Interpolation (LERP)
            current_targets = (1 - alpha) * predictions[frame_idx] + alpha * predictions[next_frame_idx]
            
            # Apply targets to actuators
            data.ctrl[:] = current_targets
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Maintain 60FPS real-time visualization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("[*] Neural Simulation Complete.")

if __name__ == "__main__":
    run_neuro_simulation()