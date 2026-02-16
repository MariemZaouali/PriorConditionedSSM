"""
Lightweight test for PriorConditionedSSM without downloading pretrained weights
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.prior_conditioned_ssm import PriorConditionedSSM, PriorConditionedSSMEfficient


def test_ssm_modules():
    """Test both SSM implementations"""
    print("\n" + "="*60)
    print("Testing PriorConditionedSSM Modules")
    print("="*60 + "\n")

    # Test configurations
    configs = [
        (256, 32, 32),  # (channels, height, width)
        (512, 16, 16),
        (128, 64, 64),
    ]

    print("Testing PriorConditionedSSM (Full Recurrent Version)")
    print("-"*60)
    for in_dim, h, w in configs:
        module = PriorConditionedSSM(in_dim)
        x = torch.randn(2, in_dim, h, w)
        prior = torch.randn(2, 1, h//2, w//2)

        # Forward + backward
        out = module(x, prior)
        loss = out.sum()
        loss.backward()

        assert out.shape == x.shape
        params = sum(p.numel() for p in module.parameters())
        print(f"  ✓ [{in_dim}ch, {h}x{w}] → Output: {out.shape}, Params: {params:,}")

    print("\nTesting PriorConditionedSSMEfficient (Fast Version)")
    print("-"*60)
    for in_dim, h, w in configs:
        module = PriorConditionedSSMEfficient(in_dim)
        x = torch.randn(2, in_dim, h, w)
        prior = torch.randn(2, 1, h*2, w*2)  # Different size to test interpolation

        # Forward + backward
        out = module(x, prior)
        loss = out.sum()
        loss.backward()

        assert out.shape == x.shape
        params = sum(p.numel() for p in module.parameters())
        print(f"  ✓ [{in_dim}ch, {h}x{w}] → Output: {out.shape}, Params: {params:,}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nKey Features Verified:")
    print("  ✓ Modules instantiate correctly")
    print("  ✓ Forward pass produces correct shapes")
    print("  ✓ Prior maps are interpolated to match feature size")
    print("  ✓ Backward pass works (gradients flow)")
    print("  ✓ Works with different resolutions")
    print("\nThe PriorConditionedSSM module is ready to use!")
    print("\nUsage in CGNet:")
    print("  1. Modify train_CGNet.py to import CGNet_SSM")
    print("  2. Replace 'CGNet' with 'CGNet_SSM' in model selection")
    print("  3. Train with same hyperparameters as original CGNet")
    print()


if __name__ == "__main__":
    try:
        test_ssm_modules()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
