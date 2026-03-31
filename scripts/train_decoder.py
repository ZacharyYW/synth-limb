"""
Syntho-Limb NS2C Engine: Sprint 2 - Distilled Decoder
=====================================================
Trains a lightweight PyTorch LSTM to map high-dimensional synthetic 
motor-intent (BA4/BA6) to 7-DOF physical robotic kinematics.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class NS2CDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=7):
        super(NS2CDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM core to handle the temporal lag between thought and action
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Linear readout layer to map hidden states to 7-DOF kinematics
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of each time-step into joint velocities
        out = self.fc(out)
        return out

def train_engine():
    print("[*] Initializing Syntho-Limb Decoder Training...")
    
    # 1. Load the Data
    X_path = "data/synthetic_fmri/motor_intent_refined.npy"
    Y_path = "data/synthetic_fmri/grasp_kinematics.npy"
    
    if not os.path.exists(X_path) or not os.path.exists(Y_path):
        print("[!] Missing data arrays. Please verify paths.")
        return
        
    X_np = np.load(X_path)  # Shape: (4, N_Vertices)
    Y_np = np.load(Y_path)  # Shape: (4, 7)
    
    # Reshape for LSTM: (Batch_Size, Sequence_Length, Features)
    # We have 1 video, so Batch Size = 1
    X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(0)
    
    input_features = X_np.shape[1]
    print(f"[*] Data loaded. Input features (Masked Voxels): {input_features}")
    
    # 2. Initialize Model, Loss, and Optimizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NS2CDecoder(input_size=input_features).to(device)
    
    X_tensor, Y_tensor = X_tensor.to(device), Y_tensor.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Training Loop (Overfitting for Proof-of-Concept)
    epochs = 150
    print(f"[*] Commencing training over {epochs} epochs on {device.type.upper()}...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, Y_tensor)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # 4. Save the trained weights
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/ns2c_decoder_v1.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"[*] SUCCESS: Trained decoder weights saved to {model_save_path}")

if __name__ == "__main__":
    train_engine()