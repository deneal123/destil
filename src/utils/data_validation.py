"""Data Validation Utilities for the SmolVLA Optimization Framework.

This module provides comprehensive data validation tools for training and optimization processes.
"""

import torch
import numpy as np
from typing import Union, List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation utilities for training and optimization processes."""
    
    def __init__(self):
        self.validation_results = {}
        self.errors_found = False
    
    def validate_tensor_shape(self, tensor: torch.Tensor, expected_shape: tuple, tensor_name: str = "tensor"):
        """Validate tensor shape against expected dimensions."""
        if tensor.shape != expected_shape:
            error_msg = f"{tensor_name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
            logger.warning(error_msg)
            self.validation_results[f"{tensor_name}_shape_valid"] = False
            self.validation_results[f"{tensor_name}_shape_error"] = error_msg
            self.errors_found = True
            return False
        else:
            self.validation_results[f"{tensor_name}_shape_valid"] = True
            return True

    def validate_tensor_range(self, tensor: torch.Tensor, min_val: float = -float('inf'), max_val: float = float('inf'), tensor_name: str = "tensor"):
        """Validate tensor values are within specified range."""
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        if tensor_min < min_val or tensor_max > max_val:
            error_msg = f"{tensor_name} values out of range: min={tensor_min}, max={tensor_max}, expected range=[{min_val}, {max_val}]"
            logger.warning(error_msg)
            self.validation_results[f"{tensor_name}_range_valid"] = False
            self.validation_results[f"{tensor_name}_range_error"] = error_msg
            self.errors_found = True
            return False
        else:
            self.validation_results[f"{tensor_name}_range_valid"] = True
            return True

    def validate_tensor_nan_inf(self, tensor: torch.Tensor, tensor_name: str = "tensor"):
        """Check for NaN and Inf values in tensor."""
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        neg_inf_count = torch.isneginf(tensor).sum().item()
        pos_inf_count = torch.isposinf(tensor).sum().item()
        
        if nan_count > 0 or inf_count > 0:
            error_msg = f"{tensor_name} contains invalid values: {nan_count} NaN, {neg_inf_count} -Inf, {pos_inf_count} +Inf"
            logger.warning(error_msg)
            self.validation_results[f"{tensor_name}_nan_inf_valid"] = False
            self.validation_results[f"{tensor_name}_nan_inf_error"] = error_msg
            self.errors_found = True
            return False
        else:
            self.validation_results[f"{tensor_name}_nan_inf_valid"] = True
            return True

    def validate_batch_consistency(self, tensors: List[torch.Tensor], tensor_names: List[str]):
        """Validate that all tensors in a batch have consistent shapes."""
        if len(tensors) != len(tensor_names):
            logger.error("Mismatch between number of tensors and tensor names")
            return False

        # Check if all tensors have the same batch size
        batch_sizes = [t.shape[0] for t in tensors]
        if not all(bs == batch_sizes[0] for bs in batch_sizes):
            error_msg = f"Inconsistent batch sizes: {[int(bs) for bs in batch_sizes]} for {tensor_names}"
            logger.warning(error_msg)
            self.validation_results["batch_consistency_valid"] = False
            self.validation_results["batch_consistency_error"] = error_msg
            self.errors_found = True
            return False

        self.validation_results["batch_consistency_valid"] = True
        return True

    def validate_data_quality_score(self, tensor: torch.Tensor, threshold: float = 0.95, tensor_name: str = "tensor"):
        """Calculate and validate data quality score based on various metrics."""
        # Calculate quality metrics
        total_elements = tensor.numel()
        nan_elements = torch.isnan(tensor).sum().item()
        inf_elements = torch.isinf(tensor).sum().item()
        zero_elements = (tensor == 0).sum().item()
        low_variance_elements = (tensor.var() < 1e-6).sum().item() if tensor.numel() > 1 else 0

        # Calculate quality score
        invalid_elements = nan_elements + inf_elements
        quality_score = (total_elements - invalid_elements) / total_elements

        self.validation_results[f"{tensor_name}_quality_score"] = quality_score
        self.validation_results[f"{tensor_name}_nan_count"] = nan_elements
        self.validation_results[f"{tensor_name}_inf_count"] = inf_elements
        self.validation_results[f"{tensor_name}_zero_ratio"] = zero_elements / total_elements

        if quality_score < threshold:
            error_msg = f"{tensor_name} quality score ({quality_score:.3f}) below threshold ({threshold})"
            logger.warning(error_msg)
            self.validation_results[f"{tensor_name}_quality_valid"] = False
            self.validation_results[f"{tensor_name}_quality_error"] = error_msg
            self.errors_found = True
            return False
        else:
            self.validation_results[f"{tensor_name}_quality_valid"] = True
            return True

    def run_comprehensive_validation(self, tensors: Dict[str, torch.Tensor], expected_shapes: Optional[Dict[str, tuple]] = None, thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        """Run comprehensive validation on a collection of tensors."""
        self.validation_results = {}
        self.errors_found = False

        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                error_msg = f"{name} is not a torch.Tensor, got {type(tensor)}"
                logger.warning(error_msg)
                self.validation_results[f"{name}_type_valid"] = False
                self.validation_results[f"{name}_type_error"] = error_msg
                self.errors_found = True
                continue

            # Validate tensor properties
            self.validate_tensor_nan_inf(tensor, name)
            self.validate_data_quality_score(tensor, tensor_name=name)

            # Validate shape if expected shape is provided
            if expected_shapes and name in expected_shapes:
                self.validate_tensor_shape(tensor, expected_shapes[name], name)

            # Validate range if thresholds are provided
            if thresholds and name in thresholds:
                range_thresholds = thresholds[name]
                min_val = range_thresholds.get('min', -float('inf'))
                max_val = range_thresholds.get('max', float('inf'))
                self.validate_tensor_range(tensor, min_val, max_val, name)

        return not self.errors_found

    def get_validation_report(self) -> Dict[str, Any]:
        """Return a comprehensive validation report."""
        return {
            'errors_found': self.errors_found,
            'validation_results': self.validation_results.copy(),
            'summary': {
                'total_checks': len(self.validation_results),
                'passed_checks': sum(1 for v in self.validation_results.values() if v is True),
                'failed_checks': sum(1 for v in self.validation_results.values() if v is False)
            }
        }


def validate_le_robot_data(img: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> Dict[str, Any]:
    """Specific validation for LeRobot dataset tensors."""
    validator = DataValidator()
    
    # Define expected shapes and ranges for LeRobot data
    expected_shapes = {
        'img': (None, 3, 224, 224),  # batch_size x channels x height x width
        'state': (None, 10),         # batch_size x state_dim
        'action': (None, 14)         # batch_size x action_dim
    }
    
    thresholds = {
        'img': {'min': -1.0, 'max': 1.0},
        'state': {'min': -10.0, 'max': 10.0},
        'action': {'min': -1.0, 'max': 1.0}
    }

    tensors = {'img': img, 'state': state, 'action': action}
    validator.run_comprehensive_validation(tensors, expected_shapes, thresholds)
    return validator.get_validation_report()


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    batch_size = 32
    img = torch.randn(batch_size, 3, 224, 224)  # Normal image tensor
    state = torch.randn(batch_size, 10)          # Normal state tensor
    action = torch.randn(batch_size, 14)         # Normal action tensor

    # Test with valid data
    print("Testing with valid data...")
    report = validate_le_robot_data(img, state, action)
    print(f"Validation passed: {not report['errors_found']}")
    print(f"Summary: {report['summary']}")

    # Test with invalid data (contains NaN)
    print("\nTesting with invalid data (NaN in action tensor)...")
    action_with_nan = action.clone()
    action_with_nan[0, 0] = float('nan')
    report = validate_le_robot_data(img, state, action_with_nan)
    print(f"Validation passed: {not report['errors_found']}")
    print(f"Summary: {report['summary']}")
    print(f"Errors: {[k for k, v in report['validation_results'].items() if 'error' in k]}")