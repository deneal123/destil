"""Model Serialization Utilities for the SmolVLA Optimization Framework.

This module provides comprehensive tools for saving, loading, and serializing models
in various formats for deployment and experimentation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Callable
import os
from pathlib import Path
import json
import logging
import hashlib
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


class ModelSerializer:
    """Comprehensive model serialization utilities for saving, loading, and converting models."""
    
    def __init__(self, output_dir: str = "serialized_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_model(self, 
                   model: nn.Module, 
                   filepath: Union[str, Path],
                   metadata: Optional[Dict[str, Any]] = None,
                   include_optimizer: Optional[torch.optim.Optimizer] = None,
                   include_config: Optional[Dict[str, Any]] = None):
        """Save a model with comprehensive metadata and optional components."""
        
        filepath = Path(filepath)
        save_dir = filepath.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare the state dict
        state_dict = {
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'model_class': model.__class__.__name__,
            'model_type': type(model).__module__
        }
        
        # Add optimizer state if provided
        if include_optimizer is not None:
            state_dict['optimizer_state_dict'] = include_optimizer.state_dict()
        
        # Add model config if provided
        if include_config is not None:
            state_dict['model_config'] = include_config
        
        # Add metadata if provided
        if metadata is not None:
            state_dict['metadata'] = metadata
        
        # Calculate hash of the model parameters for integrity check
        param_hash = self._calculate_param_hash(model.state_dict())
        state_dict['model_hash'] = param_hash
        
        # Save the model
        torch.save(state_dict, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save a human-readable metadata file
        metadata_path = filepath.with_suffix('.json')
        readable_metadata = {
            'model_path': str(filepath),
            'model_class': state_dict['model_class'],
            'timestamp': state_dict['timestamp'],
            'model_hash': param_hash,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'device': str(next(model.parameters()).device),
            'metadata': metadata or {},
            'has_optimizer': include_optimizer is not None,
            'has_config': include_config is not None
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(readable_metadata, f, indent=2)
        
        return filepath
    
    def load_model(self, 
                   model: nn.Module, 
                   filepath: Union[str, Path],
                   load_optimizer: Optional[torch.optim.Optimizer] = None,
                   strict: bool = True):
        """Load a model from a saved state dict."""
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Validate model hash if present
        if 'model_hash' in checkpoint:
            calculated_hash = self._calculate_param_hash(checkpoint['model_state_dict'])
            if calculated_hash != checkpoint['model_hash']:
                logger.warning(f"Model hash mismatch! Expected: {checkpoint['model_hash']}, Calculated: {calculated_hash}")
        
        # Load the model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state if requested and available
        if load_optimizer is not None and 'optimizer_state_dict' in checkpoint:
            load_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        
        return {
            'model_class': checkpoint.get('model_class', 'Unknown'),
            'timestamp': checkpoint.get('timestamp', 'Unknown'),
            'metadata': checkpoint.get('metadata', {}),
            'hash_validated': 'model_hash' in checkpoint
        }
    
    def save_for_deployment(self, 
                           model: nn.Module, 
                           filepath: Union[str, Path],
                           input_sample: Optional[torch.Tensor] = None,
                           opset_version: int = 11):
        """Save model in multiple formats suitable for deployment."""
        
        filepath = Path(filepath)
        save_dir = filepath.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = filepath.stem
        
        results = {}
        
        # Save as PyTorch traced model
        if input_sample is not None:
            try:
                traced_model = torch.jit.trace(model, input_sample)
                traced_path = save_dir / f"{base_name}_traced.pt"
                traced_model.save(traced_path)
                results['traced'] = str(traced_path)
                logger.info(f"Traced model saved to {traced_path}")
            except Exception as e:
                logger.error(f"Failed to save traced model: {e}")
        
        # Save as ONNX
        if input_sample is not None:
            try:
                onnx_path = save_dir / f"{base_name}.onnx"
                
                # Determine input names based on model's forward signature
                input_names = ["input"]
                if len(input_sample) == 2:  # For models expecting multiple inputs like img and state
                    input_names = ["input_img", "input_state"]
                
                torch.onnx.export(
                    model,
                    input_sample,
                    str(onnx_path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=["output"],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    } if len(input_names) == 1 else {
                        'input_img': {0: 'batch_size'},
                        'input_state': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                results['onnx'] = str(onnx_path)
                logger.info(f"ONNX model saved to {onnx_path}")
            except Exception as e:
                logger.error(f"Failed to save ONNX model: {e}")
        
        # Save as TorchScript
        try:
            scripted_model = torch.jit.script(model)
            script_path = save_dir / f"{base_name}_scripted.pt"
            scripted_model.save(script_path)
            results['scripted'] = str(script_path)
            logger.info(f"Scripted model saved to {script_path}")
        except Exception as e:
            logger.error(f"Failed to save scripted model: {e}")
        
        # Save regular checkpoint
        checkpoint_path = save_dir / f"{base_name}_checkpoint.pth"
        self.save_model(model, checkpoint_path)
        results['checkpoint'] = str(checkpoint_path)
        
        return results
    
    def _calculate_param_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Calculate a hash of model parameters for integrity checking."""
        hashes = []
        for param in state_dict.values():
            if isinstance(param, torch.Tensor):
                # Convert tensor to bytes and hash
                param_bytes = param.detach().cpu().numpy().tobytes()
                hashes.append(hashlib.sha256(param_bytes).hexdigest())
        
        # Combine all hashes
        combined_hash = hashlib.sha256(''.join(hashes).encode()).hexdigest()
        return combined_hash
    
    def validate_saved_model(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Validate a saved model file for integrity and compatibility."""
        
        filepath = Path(filepath)
        if not filepath.exists():
            return {'valid': False, 'error': f'File does not exist: {filepath}'}
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            validation_result = {
                'valid': True,
                'model_class': checkpoint.get('model_class', 'Unknown'),
                'timestamp': checkpoint.get('timestamp', 'Unknown'),
                'has_optimizer': 'optimizer_state_dict' in checkpoint,
                'has_config': 'model_config' in checkpoint,
                'has_metadata': 'metadata' in checkpoint,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024)
            }
            
            # Check if model hash exists and validate if possible
            if 'model_hash' in checkpoint:
                # For hash validation, we'd need the original model to compare against
                validation_result['hash_present'] = True
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'error': f'Failed to load model: {str(e)}'}
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive information about a model."""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buffer_params = sum(p.numel() for p in model.buffers())
        
        return {
            'class_name': model.__class__.__name__,
            'num_total_parameters': total_params,
            'num_trainable_parameters': trainable_params,
            'num_buffer_parameters': buffer_params,
            'parameter_memory_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': str(next(model.parameters()).device),
            'layers': [name for name, _ in model.named_modules()],
            'layer_types': [type(module).__name__ for _, module in model.named_modules()]
        }


def save_model_with_config(model: nn.Module, 
                          config: Dict[str, Any], 
                          filepath: Union[str, Path],
                          include_optimizer: Optional[torch.optim.Optimizer] = None):
    """Convenience function to save a model along with its configuration."""
    
    serializer = ModelSerializer()
    return serializer.save_model(
        model=model,
        filepath=filepath,
        include_config=config,
        include_optimizer=include_optimizer
    )


def load_model_with_config(model: nn.Module, 
                          filepath: Union[str, Path],
                          load_optimizer: Optional[torch.optim.Optimizer] = None):
    """Convenience function to load a model along with its configuration."""
    
    serializer = ModelSerializer()
    info = serializer.load_model(model, filepath, load_optimizer=load_optimizer)
    
    # Also try to load the config if it was saved
    checkpoint = torch.load(Path(filepath), map_location='cpu')
    if 'model_config' in checkpoint:
        info['config'] = checkpoint['model_config']
    
    return info


# Example usage and testing
if __name__ == "__main__":
    # Example usage with a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create model and test serialization
    model = SimpleModel()
    serializer = ModelSerializer()
    
    # Save model with metadata
    save_path = serializer.output_dir / "test_model.pth"
    metadata = {
        'description': 'Test model for serialization utilities',
        'author': 'SmolVLA Framework',
        'version': '1.0'
    }
    
    print("Testing model serialization...")
    serializer.save_model(model, save_path, metadata=metadata)
    
    # Load model back
    new_model = SimpleModel()
    info = serializer.load_model(new_model, save_path)
    print(f"Loaded model info: {info}")
    
    # Test model info extraction
    model_info = serializer.get_model_info(model)
    print(f"Model info: {model_info}")
    
    # Validate saved model
    validation = serializer.validate_saved_model(save_path)
    print(f"Model validation: {validation}")
    
    # Test saving for deployment
    sample_input = torch.randn(1, 10)
    deploy_results = serializer.save_for_deployment(model, serializer.output_dir / "deploy_model", input_sample=sample_input)
    print(f"Deployment save results: {deploy_results}")