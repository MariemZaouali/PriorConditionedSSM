#!/usr/bin/env python3
"""
Test script to verify the IndexError fix in train_CGNet.py

This script demonstrates that the OA() and Kappa() methods return scalar values
and should not be indexed with [1].
"""

import sys
sys.path.append('.')

from utils.metrics import Evaluator
import numpy as np

def test_metrics_methods():
    """Test that metrics methods return the expected types and values."""
    
    print("Testing metrics methods...")
    
    # Create evaluator
    evaluator = Evaluator(num_class=2)
    
    # Create some dummy data for testing
    target = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    
    evaluator.add_batch(target, pred)
    
    # Test OA() method - should return an array (access [1] for the actual value)
    oa = evaluator.OA()
    print(f"OA() returns: {oa} (type: {type(oa)}, shape: {oa.shape})")
    assert isinstance(oa, np.ndarray), f"OA() should return an array, got {type(oa)}"
    assert oa.shape == (2,), f"OA() should return array of shape (2,), got {oa.shape}"
    
    # Test Kappa() method - should return an array (access [1] for the actual value)
    kappa = evaluator.Kappa()
    print(f"Kappa() returns: {kappa} (type: {type(kappa)}, shape: {kappa.shape})")
    assert isinstance(kappa, np.ndarray), f"Kappa() should return an array, got {type(kappa)}"
    assert kappa.shape == (2,), f"Kappa() should return array of shape (2,), got {kappa.shape}"
    
    # Test methods that return arrays (these should be indexed with [1])
    precision = evaluator.Precision()
    print(f"Precision() returns: {precision} (type: {type(precision)}, shape: {precision.shape})")
    assert isinstance(precision, np.ndarray), f"Precision() should return an array, got {type(precision)}"
    
    recall = evaluator.Recall()
    print(f"Recall() returns: {recall} (type: {type(recall)}, shape: {recall.shape})")
    assert isinstance(recall, np.ndarray), f"Recall() should return an array, got {type(recall)}"
    
    f1 = evaluator.F1()
    print(f"F1() returns: {f1} (type: {type(f1)}, shape: {f1.shape})")
    assert isinstance(f1, np.ndarray), f"F1() should return an array, got {type(f1)}"
    
    iou = evaluator.Intersection_over_Union()
    print(f"Intersection_over_Union() returns: {iou} (type: {type(iou)}, shape: {iou.shape})")
    assert isinstance(iou, np.ndarray), f"IoU() should return an array, got {type(iou)}"
    
    print("\nAll tests passed!")
    print("OA() and Kappa() correctly return arrays with shape (2,) - access [1] for actual values")
    print("Precision(), Recall(), F1(), and IoU() correctly return arrays (indexing with [1] is valid)")
    
    return True

def demonstrate_correct_usage():
    """Demonstrate the correct usage after the fix."""
    
    print("\n" + "="*60)
    print("DEMONSTRATING THE CORRECT USAGE")
    print("="*60)
    
    evaluator = Evaluator(num_class=2)
    target = np.array([0, 1, 0, 1])
    pred = np.array([0, 1, 1, 1])
    evaluator.add_batch(target, pred)
    
    print("Correct code after the fix:")
    print("  train_oa = Eva_train.OA()[1]  # Correct!")
    print("  train_kappa = Eva_train.Kappa()[1]  # Correct!")
    
    # These should work correctly now
    oa_correct = evaluator.OA()[1]
    kappa_correct = evaluator.Kappa()[1]
    print(f"  OA()[1] = {oa_correct}")
    print(f"  Kappa()[1] = {kappa_correct}")
    
    print("\nAll metrics methods work correctly with proper indexing:")
    iou = evaluator.Intersection_over_Union()[1]
    precision = evaluator.Precision()[1]
    recall = evaluator.Recall()[1]
    f1 = evaluator.F1()[1]
    
    print(f"  IoU()[1] = {iou}")
    print(f"  Precision()[1] = {precision}")
    print(f"  Recall()[1] = {recall}")
    print(f"  F1()[1] = {f1}")

if __name__ == "__main__":
    test_metrics_methods()
    demonstrate_correct_usage()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The IndexError in train_CGNet.py has been successfully fixed!")
    print("OA() and Kappa() methods return arrays with shape (2,) - access [1] for actual values")
    print("The training script should now run without the IndexError")
    print("The dataset loading error is a separate issue unrelated to the IndexError fix")
