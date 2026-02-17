import torch
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla import configuration_smolvla
import json
from typing import Dict, Any, List
import os
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

class SmolVLAAnalyzer:
    
    def __init__(self, model_id: str = "lerobot/smolvla_base"):
        self.model_id = model_id
        self.model = None
        self.config = None
        self.analysis_results = {}
        
    def load_model(self):
        print(f"Analyzing model: {self.model_id}")

        try:
            self.config = configuration_smolvla.SmolVLAConfig()
            print("Configuration loaded successfully")
        except Exception as e:
            print(f"Could not load configuration: {e}")
            self.config = {}
            
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Attempting to load pretrained model...")
            self.model_loaded = False
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model_loaded = False
            
        return self.model_loaded
        
    def analyze_architecture(self) -> Dict[str, Any]:
        if self.config is None:
            self.load_model()
            
        analysis = {
            "model_info": {
                "model_id": self.model_id,
                "config_available": self.config is not None
            },
            "architecture_components": {},
            "parameter_estimates": {},
            "configuration_details": {}
        }

        if hasattr(self.config, '__dict__'):
            config_dict = {k: v for k, v in self.config.__dict__.items() 
                          if not k.startswith('_') and not callable(v)}
            analysis["configuration_details"] = config_dict

            key_params = [
                'hidden_size', 'num_attention_heads', 'num_hidden_layers',
                'image_size', 'patch_size', 'max_position_embeddings',
                'intermediate_size', 'vocab_size'
            ]
            
            for param in key_params:
                if param in config_dict:
                    analysis["configuration_details"][param] = config_dict[param]
            
            self._estimate_parameters(analysis)
            
        self.analysis_results = analysis
        return analysis
    
    def _estimate_parameters(self, analysis: Dict[str, Any]):
        config = analysis["configuration_details"]
        
        hidden_size = config.get('hidden_size', 768)
        num_heads = config.get('num_attention_heads', 12)
        num_layers = config.get('num_hidden_layers', 12)
        intermediate_size = config.get('intermediate_size', 3072)
        vocab_size = config.get('vocab_size', 30522)
        image_size = config.get('image_size', 224)
        patch_size = config.get('patch_size', 16)
        
        num_patches = (image_size // patch_size) ** 2
        vision_encoder_params = (
            # Patch embedding
            3 * patch_size * patch_size * hidden_size +
            # Position embeddings
            (num_patches + 1) * hidden_size +
            # Layer normalization
            2 * hidden_size
        )
        
        attention_params_per_layer = (
            # QKV projection
            3 * hidden_size * hidden_size +
            # Output projection
            hidden_size * hidden_size +
            # Layer norm
            2 * hidden_size
        )
        
        ffn_params_per_layer = (
            # First linear
            hidden_size * intermediate_size +
            # Second linear
            intermediate_size * hidden_size +
            # Layer norm
            2 * hidden_size
        )
        
        transformer_params = num_layers * (attention_params_per_layer + ffn_params_per_layer)

        action_head_params = (
            hidden_size * 1024 +  # Hidden to intermediate
            1024 * 512 +          # Intermediate layer
            512 * 7               # Output (assuming 7 DOF robot)
        )

        total_params = vision_encoder_params + transformer_params + action_head_params
        
        analysis["parameter_estimates"] = {
            "vision_encoder": vision_encoder_params,
            "transformer": transformer_params,
            "action_head": action_head_params,
            "total": total_params
        }
        
    def print_summary(self):
        if not self.analysis_results:
            self.analyze_architecture()
                
        print("\n" + "="*60)
        print("SMOLVLA MODEL ANALYSIS SUMMARY")
        print("="*60)
            
        print(f"\nModel: {self.analysis_results['model_info']['model_id']}")
            
        if 'parameter_estimates' in self.analysis_results:
            total_params = self.analysis_results['parameter_estimates']['total']
            print(f"Estimated Total Parameters: {total_params:,}")
                
            print("\nComponent Breakdown (Estimates):")
            for component, params in self.analysis_results['parameter_estimates'].items():
                if component != 'total' and params > 0:
                    percentage = (params / total_params) * 100
                    print(f"  {component}: {params:,} parameters ({percentage:.1f}%)")
            
        if 'configuration_details' in self.analysis_results:
            print("\nKey Configuration Details:")
            important_keys = ['hidden_size', 'num_attention_heads', 'num_hidden_layers', 
                            'image_size', 'patch_size', 'max_position_embeddings',
                            'intermediate_size', 'vocab_size']
            for key in important_keys:
                if key in self.analysis_results['configuration_details']:
                    value = self.analysis_results['configuration_details'][key]
                    print(f"  {key}: {value}")
                        
        print("\nNote: Parameter counts are estimates based on configuration.")
        print("Actual counts may vary based on specific implementation details.")
        
    def save_analysis(self, filepath: str = "results/model_analysis.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        with open(filepath, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"Analysis saved to: {filepath}")

    def count_actual_parameters(self, model) -> Dict[str, int]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        layer_params = {}
        for name, param in model.named_parameters():
            layer_params[name] = param.numel()
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'per_layer': layer_params
        }
    
    def estimate_memory_footprint(self, model, input_shape=(1, 3, 224, 224)) -> Dict[str, float]:
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
        
        try:
            dummy_input = torch.randn(*input_shape)
            with torch.no_grad():

                activation_sizes = []
                hooks = []
                
                def register_hook(module):
                    def hook_fn(module_, input_, output_):
                        if isinstance(output_, torch.Tensor):
                            activation_sizes.append(output_.numel() * output_.element_size())
                        elif isinstance(output_, (list, tuple)):
                            for o in output_:
                                if isinstance(o, torch.Tensor):
                                    activation_sizes.append(o.numel() * o.element_size())
                    
                    hooks.append(module.register_forward_hook(hook_fn))
                
                model.apply(register_hook)

                _ = model(dummy_input)

                for hook in hooks:
                    hook.remove()
                
                activation_memory = sum(activation_sizes) / (1024**2)  # MB
        except:
            activation_memory = param_memory * 2  # Rough estimate
        
        return {
            'parameter_memory_mb': param_memory,
            'activation_memory_mb': activation_memory,
            'estimated_total_memory_mb': param_memory + activation_memory
        }
    
    def analyze_computational_complexity(self, model, input_shape=(1, 3, 224, 224)) -> Dict[str, Any]:
        try:
            from thop import profile
            dummy_input = torch.randn(*input_shape)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            return {
                'flops': flops,
                'params': params,
                'flops_giga': flops / 1e9,
                'params_million': params / 1e6
            }
        except ImportError:
            logger.warning("thop not installed. Install with 'pip install thop' for FLOPs estimation.")
            params = sum(p.numel() for p in model.parameters())
            return {
                'flops_estimate': params * 3,  # Very rough approximation
                'params': params,
                'flops_giga_estimate': (params * 3) / 1e9,
                'params_million': params / 1e6,
                'note': 'Install thop for accurate FLOPs estimation'
            }
    
    def check_model_compatibility(self, model, test_inputs) -> Dict[str, bool]:
        compatibility_results = {}
        
        for input_name, input_tensor in test_inputs.items():
            try:
                with torch.no_grad():
                    output = model(input_tensor)
                    compatibility_results[input_name] = {
                        'compatible': True,
                        'output_shape': output.shape if isinstance(output, torch.Tensor) else 'variable',
                        'success': True
                    }
            except Exception as e:
                compatibility_results[input_name] = {
                    'compatible': False,
                    'error': str(e),
                    'success': False
                }
        
        return compatibility_results
    
    def visualize_architecture(self, model, save_path: str = None) -> None:
        try:
            import torchview
            dummy_input = torch.randn(1, 3, 224, 224)
            model_graph = torchview.draw_graph(model, input_data=dummy_input, expand_nested=True)
            if save_path:
                model_graph.visual_graph.render(save_path, format='png', cleanup=True)
            else:
                model_graph.visual_graph.view()
        except ImportError:
            logger.warning("torchview not installed. Install with 'pip install torchview' for architecture visualization.")
            print(model)
    
    def compare_models(self, models: Dict[str, torch.nn.Module], input_shape=(1, 3, 224, 224)) -> Dict[str, Dict[str, Any]]:
        comparison_results = {}
        dummy_input = torch.randn(*input_shape)
        
        for name, model in models.items():
            with torch.no_grad():
                try:
                    output = model(dummy_input)
                    params = sum(p.numel() for p in model.parameters())
                    
                    comparison_results[name] = {
                        'parameters': params,
                        'output_shape': output.shape,
                        'memory_estimate_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
                        'success': True
                    }
                except Exception as e:
                    comparison_results[name] = {
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'error': str(e),
                        'success': False
                    }
        
        return comparison_results
    
    def export_analysis_results(self, analysis_results: Dict[str, Any], export_path: str) -> None:
        import json
        import pickle
        
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export as JSON
        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Export as pickle
        pickle_path = path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(analysis_results, f)
        
        logger.info(f"Analysis results exported to {json_path} and {pickle_path}")
    
    def predict_performance(self, model, hardware_specs: Dict[str, Any] = None) -> Dict[str, float]:
        if hardware_specs is None:
            hardware_specs = {
                'cpu_cores': os.cpu_count(),
                'gpu_available': torch.cuda.is_available(),
                'memory_gb': 16  # Assume 16GB RAM
            }
        
        params = sum(p.numel() for p in model.parameters())
        estimated_flops = params * 3  # Rough estimate
        
        cpu_prediction = max(1.0, 1000000000 / estimated_flops)  # Inverse relationship
        gpu_prediction = cpu_prediction * 5 if hardware_specs['gpu_available'] else cpu_prediction  # GPU is typically faster
        
        return {
            'estimated_inference_speed_fps': min(100, max(0.1, gpu_prediction)),
            'estimated_latency_ms': 1000 / min(100, max(0.1, gpu_prediction)) if gpu_prediction > 0 else 10000,
            'memory_requirement_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
            'estimated_power_consumption_w': params / 1e6 * 2 + 5  # Rough estimate
        }
    
    def identify_bottlenecks(self, model, input_shape=(1, 3, 224, 224)) -> Dict[str, Any]:
        bottlenecks = {
            'high_param_layers': [],
            'potential_memory_bottlenecks': [],
            'potential_compute_bottlenecks': []
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                param_count = module.weight.numel()

                if param_count > 100000:  # More than 100k parameters
                    bottlenecks['high_param_layers'].append({
                        'layer_name': name,
                        'param_count': param_count
                    })

                if isinstance(module, torch.nn.Linear) and module.in_features > 1024:
                    bottlenecks['potential_compute_bottlenecks'].append({
                        'layer_name': name,
                        'type': 'Linear',
                        'input_features': module.in_features,
                        'output_features': module.out_features
                    })
                elif isinstance(module, torch.nn.Conv2d) and module.in_channels * module.out_channels > 10000:
                    bottlenecks['potential_compute_bottlenecks'].append({
                        'layer_name': name,
                        'type': 'Conv2d',
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size
                    })
        
        return bottlenecks
    
    def recommend_optimizations(self, model, target_hardware: str = 'general') -> List[Dict[str, str]]:
        recommendations = []

        param_count = sum(p.numel() for p in model.parameters())
        layer_count = len(list(model.modules()))

        if param_count > 100000000:  # 100M+ parameters
            recommendations.append({
                'strategy': 'Quantization',
                'reason': f'Model has {param_count/1e6:.1f}M parameters, quantization can reduce size and improve speed',
                'suggested_method': 'INT8 or INT4 quantization'
            })
        
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        if len(linear_layers) > 3:
            recommendations.append({
                'strategy': 'Structured Pruning',
                'reason': f'Model has {len(linear_layers)} linear layers suitable for structured pruning',
                'suggested_method': 'Magnitude-based pruning'
            })
        
        if target_hardware == 'mobile':
            recommendations.append({
                'strategy': 'Mobile Optimization',
                'reason': 'Targeting mobile deployment',
                'suggested_method': 'TensorRT optimization, INT8 quantization, model compression'
            })
        elif target_hardware == 'edge':
            recommendations.append({
                'strategy': 'Edge Optimization',
                'reason': 'Targeting edge deployment',
                'suggested_method': 'ONNX optimization, quantization, pruning'
            })
        
        if param_count > 50000000:  # 50M+ parameters
            recommendations.append({
                'strategy': 'Knowledge Distillation',
                'reason': f'Model is large ({param_count/1e6:.1f}M parameters), distillation can create efficient student model',
                'suggested_method': 'Temperature scaling, soft/hard loss combination'
            })
        
        return recommendations

if __name__ == "__main__":
    analyzer = SmolVLAAnalyzer()
    analyzer.load_model()
    analysis = analyzer.analyze_architecture()
    analyzer.print_summary()
    analyzer.save_analysis()

