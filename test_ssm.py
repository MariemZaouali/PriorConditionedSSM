"""
Test script for PriorConditionedSSM module

This script verifies:
1. Module can be instantiated correctly
2. Forward pass works with dummy inputs
3. Output shapes match input shapes
4. No gradient errors during backward pass
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.prior_conditioned_ssm import PriorConditionedSSM, PriorConditionedSSMEfficient
from network.CGNet import CGNet_SSM


def test_prior_conditioned_ssm():
    """Test PriorConditionedSSM module"""
    print("=" * 60)
    print("Testing PriorConditionedSSM Module")
    print("=" * 60)

    # Create module
    in_dim = 256
    module = PriorConditionedSSM(in_dim)
    print(f"✓ Module instantiated with in_dim={in_dim}")

    # Create dummy inputs
    batch_size = 2
    height, width = 32, 32
    x = torch.randn(batch_size, in_dim, height, width)
    guiding_map = torch.randn(batch_size, 1, height // 2, width // 2)  # Different size

    print(f"✓ Input shapes: x={x.shape}, guiding_map={guiding_map.shape}")

    # Forward pass
    output = module(x, guiding_map)
    print(f"✓ Forward pass successful")
    print(f"✓ Output shape: {output.shape}")

    # Check shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print(f"✓ Output shape matches input shape")

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print(f"✓ Backward pass successful")

    # Count parameters
    num_params = sum(p.numel() for p in module.parameters())
    print(f"✓ Number of parameters: {num_params:,}")

    print("\n✅ PriorConditionedSSM test passed!\n")


def test_prior_conditioned_ssm_efficient():
    """Test PriorConditionedSSMEfficient module"""
    print("=" * 60)
    print("Testing PriorConditionedSSMEfficient Module")
    print("=" * 60)

    # Create module
    in_dim = 512
    module = PriorConditionedSSMEfficient(in_dim)
    print(f"✓ Module instantiated with in_dim={in_dim}")

    # Create dummy inputs
    batch_size = 2
    height, width = 16, 16
    x = torch.randn(batch_size, in_dim, height, width)
    guiding_map = torch.randn(batch_size, 1, height * 4, width * 4)  # Larger size

    print(f"✓ Input shapes: x={x.shape}, guiding_map={guiding_map.shape}")

    # Forward pass
    output = module(x, guiding_map)
    print(f"✓ Forward pass successful")
    print(f"✓ Output shape: {output.shape}")

    # Check shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print(f"✓ Output shape matches input shape")

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print(f"✓ Backward pass successful")

    # Count parameters
    num_params = sum(p.numel() for p in module.parameters())
    print(f"✓ Number of parameters: {num_params:,}")

    print("\n✅ PriorConditionedSSMEfficient test passed!\n")


def test_cgnet_ssm():
    """Test full CGNet_SSM network"""
    print("=" * 60)
    print("Testing CGNet_SSM Network")
    print("=" * 60)

    # Create network
    model = CGNet_SSM()
    print(f"✓ CGNet_SSM network instantiated")

    # Create dummy inputs
    batch_size = 1
    height, width = 256, 256
    img_A = torch.randn(batch_size, 3, height, width)
    img_B = torch.randn(batch_size, 3, height, width)

    print(f"✓ Input shapes: A={img_A.shape}, B={img_B.shape}")

    # Forward pass
    change_map, final_map = model(img_A, img_B)
    print(f"✓ Forward pass successful")
    print(f"✓ Output shapes: change_map={change_map.shape}, final_map={final_map.shape}")

    # Check shapes
    assert change_map.shape == (batch_size, 1, height, width)
    assert final_map.shape == (batch_size, 1, height, width)
    print(f"✓ Output shapes are correct")

    # Test backward pass
    loss = (change_map.sum() + final_map.sum())
    loss.backward()
    print(f"✓ Backward pass successful")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total number of parameters: {num_params:,}")

    print("\n✅ CGNet_SSM test passed!\n")


def compare_modules():
    """Compare CGM and PriorConditionedSSM parameter counts"""
    print("=" * 60)
    print("Comparing Module Complexity")
    print("=" * 60)

    from network.CGNet import ChangeGuideModule

    for in_dim in [256, 512]:
        cgm = ChangeGuideModule(in_dim)
        ssm = PriorConditionedSSMEfficient(in_dim)

        cgm_params = sum(p.numel() for p in cgm.parameters())
        ssm_params = sum(p.numel() for p in ssm.parameters())

        print(f"\nDimension: {in_dim}")
        print(f"  CGM parameters:      {cgm_params:,}")
        print(f"  SSM parameters:      {ssm_params:,}")
        print(f"  Ratio (SSM/CGM):     {ssm_params/cgm_params:.2f}x")

    print("\n✅ Comparison complete!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PriorConditionedSSM Module Test Suite")
    print("=" * 60 + "\n")

    try:
        # Run all tests
        test_prior_conditioned_ssm()
        test_prior_conditioned_ssm_efficient()
        test_cgnet_ssm()
        compare_modules()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nThe PriorConditionedSSM module is ready to use!")
        print("\nTo use in training:")
        print("  from network.CGNet import CGNet_SSM")
        print("  model = CGNet_SSM()")
        print("\nOr use it standalone:")
        print("  from network.prior_conditioned_ssm import PriorConditionedSSMEfficient")
        print("  ssm = PriorConditionedSSMEfficient(in_dim=256)")
        print()

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
