import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def quantization_analysis(model, bits=8):
    try:
        logger.info(f"Analyzing model for {bits}-bit quantization...")
        
        quantizable_layers = 0
        total_layers = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                quantizable_layers += 1
            total_layers += 1
        
        quantization_feasibility = quantizable_layers / max(total_layers, 1)
        logger.info(f"Quantization feasibility: {quantization_feasibility:.2%} layers quantizable")
        
        return {
            'feasibility': quantization_feasibility,
            'quantizable_layers': quantizable_layers,
            'total_layers': total_layers
        }
    except Exception as e:
        logger.warning(f"Quantization analysis failed: {e}")
        return {'feasibility': 0.0}


def pruning_analysis(model, sparsity=0.3):
    try:
        logger.info(f"Analyzing model for {sparsity*100}% pruning...")
        
        param_count = 0
        prunable_params = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                param_count += param.numel()
                threshold = torch.quantile(torch.abs(param.data), 1 - sparsity)
                prunable_params += (torch.abs(param.data) < threshold).sum().item()
        
        pruning_potential = prunable_params / max(param_count, 1)
        logger.info(f"Pruning potential: {pruning_potential:.2%} parameters removable")
        
        return {
            'potential': pruning_potential,
            'prunable_params': prunable_params,
            'total_params': param_count
        }
    except Exception as e:
        logger.warning(f"Pruning analysis failed: {e}")
        return {'potential': 0.0}


class OptimizationManager:
    
    def __init__(self):
        self.optimization_results = {}
    
    def apply_quantization(self, model, dataloader, device, num_calibration_batches=10):
        from ..models.student import quantize_model
        return quantize_model(model, dataloader, device, num_calibration_batches)
    
    def apply_pruning(self, model, sparsity=0.2):
        from .trainer import apply_structured_pruning
        return apply_structured_pruning(model, sparsity)
    
    def apply_attention_pruning(self, model, sparsity=0.3):
        from .trainer import apply_attention_head_pruning
        return apply_attention_head_pruning(model, sparsity)
    
    def analyze_model(self, model):
        quant_results = quantization_analysis(model)
        prune_results = pruning_analysis(model)
        
        analysis = {
            'quantization': quant_results,
            'pruning': prune_results,
            'total_params': sum(p.numel() for p in model.parameters())
        }
        
        return analysis
