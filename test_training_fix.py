#!/usr/bin/env python3
"""
Test script to verify the in-place operation fix in CGNet_SSM training.
This script runs a minimal training loop to ensure the fix works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.CGNet_SSM import CGNet_SSM
import sys
import os

def test_training_fix():
    """Test that CGNet_SSM training works without in-place operation errors"""
    
    print("Testing CGNet_SSM training fix...")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CGNet_SSM().to(device)
    print("Model created successfully")
    
    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0025)
    criterion = nn.BCEWithLogitsLoss()
    
    # Enable anomaly detection to catch any remaining in-place operations
    torch.autograd.set_detect_anomaly(True)
    
    # Create dummy training data
    batch_size = 2
    height, width = 256, 256
    
    # Generate random images and binary masks
    A = torch.randn(batch_size, 3, height, width).to(device)
    B = torch.randn(batch_size, 3, height, width).to(device)
    mask = torch.randint(0, 2, (batch_size, height, width)).float().to(device)
    
    print(f"Created dummy data: A={A.shape}, B={B.shape}, mask={mask.shape}")
    
    # Training loop test
    model.train()
    total_loss = 0
    num_batches = 5
    
    try:
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(A, B)
            pred0 = preds[0].float()
            pred1 = preds[1].float()
            
            # Prepare target
            Y = mask.unsqueeze(1)  # Add channel dimension
            
            # Compute loss
            loss = criterion(pred0, Y) + criterion(pred1, Y)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"Batch {batch_idx + 1}/{num_batches}: Loss = {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        print(f"Training test completed successfully!")
        print(f"  Average loss: {avg_loss:.6f}")
        print(f"  Final output shapes: {preds[0].shape}, {preds[1].shape}")
        
        return True
        
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"In-place operation error still present: {e}")
            return False
        else:
            print(f"Other training error: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("CGNet_SSM Training Fix Verification")
    print("=" * 50)
    
    success = test_training_fix()
    
    if success:
        print("\nSUCCESS: CGNet_SSM training fix verified!")
        print("The in-place operation error has been resolved.")
        print("You can now run your training script without the RuntimeError.")
    else:
        print("\nFAILURE: Training fix verification failed.")
        print("The in-place operation error may still be present.")
        sys.exit(1)

if __name__ == "__main__":
    main()