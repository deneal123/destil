"""Experiment Tracking Utilities for the SmolVLA Optimization Framework.

This module provides comprehensive tools for tracking experiments, logging metrics,
and managing experiment results for reproducibility and analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pickle
import yaml
import logging
import hashlib
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration dataclass for experiment tracking."""
    experiment_name: str
    model_type: str
    dataset: str
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
    temperature: float = 3.0
    alpha: float = 0.7
    mixed_precision: bool = False
    grad_accum_steps: int = 1
    early_stopping_patience: Optional[int] = None
    additional_params: Optional[Dict[str, Any]] = None


class ExperimentTracker:
    """Comprehensive experiment tracking utilities for logging metrics and managing results."""
    
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_experiment = None
        self.experiment_logs = {}
        self.metrics_history = {}
    
    def start_experiment(self, config: ExperimentConfig, experiment_id: Optional[str] = None):
        """Start a new experiment with the given configuration."""
        
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{config.experiment_name}_{timestamp}"
        
        self.current_experiment = experiment_id
        self.experiment_dir = self.output_dir / experiment_id
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        config_dict = asdict(config)
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Initialize metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        # Initialize logs
        self.experiment_logs = {
            'experiment_id': experiment_id,
            'start_time': datetime.now().isoformat(),
            'config': config_dict,
            'metrics': self.metrics_history.copy(),
            'artifacts': []
        }
        
        logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_metrics(self, epoch: int, train_loss: float, val_loss: float, learning_rate: float = None, additional_metrics: Optional[Dict[str, float]] = None):
        """Log metrics for the current epoch."""
        
        if self.current_experiment is None:
            raise ValueError("No experiment is currently active. Call start_experiment first.")
        
        # Update metrics history
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        if learning_rate is not None:
            self.metrics_history['learning_rate'].append(learning_rate)
        
        # Log additional metrics if provided
        if additional_metrics:
            for key, value in additional_metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)
        
        # Update experiment logs
        self.experiment_logs['metrics'] = self.metrics_history.copy()
        
        # Save metrics to file
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.debug(f"Logged metrics for epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    def log_artifact(self, artifact_path: Union[str, Path], artifact_name: str = None):
        """Log an artifact (model, plot, etc.) to the experiment."""
        
        if self.current_experiment is None:
            raise ValueError("No experiment is currently active. Call start_experiment first.")
        
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning(f"Artifact does not exist: {artifact_path}")
            return
        
        if artifact_name is None:
            artifact_name = artifact_path.name
        
        # Copy artifact to experiment directory
        dest_path = self.experiment_dir / f"artifact_{artifact_name}"
        dest_path.write_bytes(artifact_path.read_bytes())
        
        # Record in logs
        artifact_info = {
            'name': artifact_name,
            'path': str(dest_path),
            'size': artifact_path.stat().st_size,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experiment_logs['artifacts'].append(artifact_info)
        
        logger.info(f"Logged artifact: {artifact_name}")
    
    def log_system_info(self):
        """Log system information for reproducibility."""
        
        if self.current_experiment is None:
            raise ValueError("No experiment is currently active. Call start_experiment first.")
        
        import platform
        import psutil
        import torch
        
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'cpu_count': os.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count(),
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            system_info['gpu_details'] = {
                'device_name': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        
        # Save system info
        system_info_path = self.experiment_dir / "system_info.json"
        with open(system_info_path, 'w') as f:
            json.dump(system_info, f, indent=2)
        
        self.experiment_logs['system_info'] = system_info
        logger.info("Logged system information for reproducibility")
    
    def end_experiment(self, final_metrics: Optional[Dict[str, float]] = None, notes: str = ""):
        """End the current experiment and save final results."""
        
        if self.current_experiment is None:
            raise ValueError("No experiment is currently active. Call start_experiment first.")
        
        # Add final metrics if provided
        if final_metrics:
            self.experiment_logs['final_metrics'] = final_metrics
        
        # Add notes if provided
        if notes:
            self.experiment_logs['notes'] = notes
        
        # Add end time
        self.experiment_logs['end_time'] = datetime.now().isoformat()
        
        # Calculate experiment duration
        start_time = datetime.fromisoformat(self.experiment_logs['start_time'])
        end_time = datetime.fromisoformat(self.experiment_logs['end_time'])
        duration = (end_time - start_time).total_seconds()
        self.experiment_logs['duration_seconds'] = duration
        
        # Save complete experiment log
        log_path = self.experiment_dir / "experiment_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.experiment_logs, f, indent=2, default=str)
        
        # Create a summary file
        summary = {
            'experiment_id': self.current_experiment,
            'experiment_name': self.experiment_logs['config']['experiment_name'],
            'duration_minutes': round(duration / 60, 2),
            'epochs_completed': len(self.metrics_history['train_loss']),
            'final_train_loss': self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else None,
            'final_val_loss': self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else None,
            'best_val_loss': min(self.metrics_history['val_loss']) if self.metrics_history['val_loss'] else None,
            'artifacts_count': len(self.experiment_logs['artifacts'])
        }
        
        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Ended experiment: {self.current_experiment}")
        logger.info(f"Results saved to: {self.experiment_dir}")
        
        # Clear current experiment
        experiment_id = self.current_experiment
        self.current_experiment = None
        return summary
    
    def get_experiment_history(self) -> List[Dict[str, Any]]:
        """Get a list of all tracked experiments."""
        
        experiments = []
        for exp_dir in self.output_dir.iterdir():
            if exp_dir.is_dir():
                summary_path = exp_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        experiments.append(summary)
        
        # Sort by start time
        experiments.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple experiments."""
        
        comparison = {
            'experiments': {},
            'comparison_metrics': ['final_train_loss', 'final_val_loss', 'best_val_loss', 'duration_minutes']
        }
        
        for exp_id in experiment_ids:
            exp_dir = self.output_dir / exp_id
            summary_path = exp_dir / "summary.json"
            
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    comparison['experiments'][exp_id] = summary
        
        return comparison


class MLflowExperimentTracker:
    """MLflow-compatible experiment tracker for advanced tracking."""
    
    def __init__(self, tracking_uri: str = None):
        self.mlflow_available = self._check_mlflow()
        self.tracking_uri = tracking_uri
        self.active_run = None
        
        if self.mlflow_available:
            import mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
    
    def _check_mlflow(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow
            return True
        except ImportError:
            logger.warning("MLflow not installed. Install with 'pip install mlflow' for advanced tracking.")
            return False
    
    def start_mlflow_experiment(self, experiment_name: str, run_name: str = None):
        """Start an MLflow experiment run."""
        
        if not self.mlflow_available:
            logger.warning("MLflow not available. Cannot start MLflow experiment.")
            return
        
        import mlflow
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        self.active_run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow experiment run: {self.active_run.info.run_id}")
    
    def log_mlflow_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        
        if not self.mlflow_available or not self.active_run:
            logger.warning("MLflow not available or no active run. Cannot log parameters.")
            return
        
        import mlflow
        mlflow.log_params(params)
    
    def log_mlflow_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow."""
        
        if not self.mlflow_available or not self.active_run:
            logger.warning("MLflow not available or no active run. Cannot log metrics.")
            return
        
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    
    def log_mlflow_artifact(self, local_path: str, artifact_path: str = None):
        """Log an artifact to MLflow."""
        
        if not self.mlflow_available or not self.active_run:
            logger.warning("MLflow not available or no active run. Cannot log artifact.")
            return
        
        import mlflow
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_mlflow_experiment(self):
        """End the active MLflow experiment run."""
        
        if not self.mlflow_available or not self.active_run:
            logger.warning("MLflow not available or no active run.")
            return
        
        import mlflow
        mlflow.end_run()
        self.active_run = None
        logger.info("Ended MLflow experiment run.")


def create_experiment_config(experiment_name: str, model_type: str, dataset: str, **kwargs) -> ExperimentConfig:
    """Convenience function to create an experiment configuration."""
    
    # Set defaults for required parameters if not provided
    config_params = {
        'experiment_name': experiment_name,
        'model_type': model_type,
        'dataset': dataset,
        'optimizer': kwargs.get('optimizer', 'AdamW'),
        'learning_rate': kwargs.get('learning_rate', 1e-4),
        'batch_size': kwargs.get('batch_size', 32),
        'epochs': kwargs.get('epochs', 10),
        'temperature': kwargs.get('temperature', 3.0),
        'alpha': kwargs.get('alpha', 0.7),
        'mixed_precision': kwargs.get('mixed_precision', False),
        'grad_accum_steps': kwargs.get('grad_accum_steps', 1),
        'early_stopping_patience': kwargs.get('early_stopping_patience', None),
        'additional_params': kwargs.get('additional_params', {})
    }
    
    return ExperimentConfig(**config_params)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Testing experiment tracking utilities...")
    
    # Create a sample experiment configuration
    config = create_experiment_config(
        experiment_name="test_distillation",
        model_type="StudentModel",
        dataset="LeRobot",
        learning_rate=1e-4,
        batch_size=16,
        epochs=5,
        temperature=3.0,
        alpha=0.7
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(output_dir="test_experiments")
    
    # Start experiment
    exp_id = tracker.start_experiment(config)
    
    # Log some sample metrics
    for epoch in range(1, 6):
        train_loss = 1.0 - epoch * 0.1  # Simulated decreasing loss
        val_loss = 1.1 - epoch * 0.09   # Simulated decreasing loss
        tracker.log_metrics(epoch, train_loss, val_loss, learning_rate=1e-4)
    
    # End experiment
    final_metrics = {
        'accuracy': 0.85,
        'f1_score': 0.82,
        'best_epoch': 4
    }
    summary = tracker.end_experiment(final_metrics=final_metrics, notes="Test experiment completed successfully")
    
    print(f"Experiment summary: {summary}")
    
    # List all experiments
    experiments = tracker.get_experiment_history()
    print(f"All experiments: {len(experiments)}")
    
    # Test MLflow tracker (will show warning if MLflow not installed)
    mlflow_tracker = MLflowExperimentTracker()
    print("MLflow tracker initialized")