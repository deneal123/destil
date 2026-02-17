"""Real SmolVLA training with HuggingFace integration, profiling, and full optimization pipeline."""

import argparse
import os
import logging
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io._video_deprecation_warning")
warnings.filterwarnings("ignore", message=".*torchcodec.*")

from tqdm import tqdm

from src.models.teacher import RealSmolVLAModel
from src.models.student import StudentModel, quantize_model
from src.datasets.real_dataset import RealLeRobotDataset, create_dataloader
from src.training.trainer import DistillationTrainer, OptimizationProfiler, apply_structured_pruning, apply_attention_head_pruning
from src.training.optimizers import OptimizationManager



HF_DATASETS_CACHE = os.getenv('HF_DATASETS_CACHE', os.path.expanduser('~/.cache/huggingface/datasets'))
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
DATASET_DOWNLOAD_PERCENTAGE = float(os.getenv('DATASET_DOWNLOAD_PERCENTAGE', '1.0'))

PROFILING_AVAILABLE = True
AMP_AVAILABLE = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

QUANT_AVAILABLE = True



def main():
    parser = argparse.ArgumentParser(description='Real SmolVLA Optimization Pipeline')
    parser.add_argument('--model_id', type=str, default='lerobot/smolvla_base')
    parser.add_argument('--dataset', type=str, default='lerobot/libero')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--student_ratio', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--quantize', action='store_true', help='Analyze quantization')
    parser.add_argument('--prune', action='store_true', help='Analyze pruning')
    parser.add_argument('--dataset_cache_dir', type=str, default=None, 
                       help='Custom dataset cache directory (overrides HF_DATASETS_CACHE env var)')
    parser.add_argument('--dataset_percentage', type=float, default=None, 
                       help='Percentage of dataset to use (0.0-1.0, overrides DATASET_DOWNLOAD_PERCENTAGE env var)')
    parser.add_argument('--output_dir', type=str, default='results/real_optimization')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.profile:
        os.makedirs(os.path.join(args.output_dir, 'profiles'), exist_ok=True)
    
    if args.dataset_cache_dir:
        os.environ['HF_DATASETS_CACHE'] = args.dataset_cache_dir
        global HF_DATASETS_CACHE
        HF_DATASETS_CACHE = args.dataset_cache_dir
    
    if args.dataset_percentage is not None:
        global DATASET_DOWNLOAD_PERCENTAGE
        DATASET_DOWNLOAD_PERCENTAGE = args.dataset_percentage
    
    logger.info("=" * 60)
    logger.info("REAL SMOLVLA OPTIMIZATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Dataset cache: {HF_DATASETS_CACHE}")
    logger.info(f"Dataset percentage: {DATASET_DOWNLOAD_PERCENTAGE*100:.1f}%")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info(f"Profiling: {args.profile}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading teacher model...")
    teacher = RealSmolVLAModel(args.model_id, use_real=True).to(device)
    
    logger.info("Creating student model...")
    student = StudentModel(ratio=args.student_ratio).to(device)
    
    teacher_params = teacher.get_num_parameters()
    student_params = student.get_num_parameters()
    
    logger.info(f"Teacher params: {teacher_params:,}")
    logger.info(f"Student params: {student_params:,}")
    logger.info(f"Compression ratio: {teacher_params/student_params:.1f}x")
    
    if args.profile:
        profiler = OptimizationProfiler(os.path.join(args.output_dir, 'profiles'))
        logger.info("Profiling teacher model...")
        teacher_metrics = profiler.profile_model(teacher, device=device)
        logger.info(f"Teacher latency: {teacher_metrics['latency']['mean']:.2f}±{teacher_metrics['latency']['std']:.2f} ms")
        logger.info(f"Teacher memory: {teacher_metrics['memory_mb']:.1f} MB")
        
        logger.info("Profiling student model...")
        student_metrics = profiler.profile_model(student, device=device)
        logger.info(f"Student latency: {student_metrics['latency']['mean']:.2f}±{student_metrics['latency']['std']:.2f} ms")
        logger.info(f"Student memory: {student_metrics['memory_mb']:.1f} MB")
    
    optimization_manager = OptimizationManager()
    if args.quantize or args.prune:
        analysis_results = optimization_manager.analyze_model(teacher)
        if args.quantize:
            quant_results = analysis_results['quantization']
            logger.info(f"Quantization feasibility: {quant_results['feasibility']:.1%}")
        
        if args.prune:
            prune_results = analysis_results['pruning']
            logger.info(f"Pruning potential: {prune_results['potential']:.1%}")
    
    logger.info(f"Loading dataset: {args.dataset}")
    train_dataset = RealLeRobotDataset(args.dataset, 'train', args.num_samples)
    val_dataset = RealLeRobotDataset(args.dataset, 'val', args.num_samples // 5)
    
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, num_workers=2)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        temperature=args.temperature,
        alpha=args.alpha,
        device=device
    )
    
    best_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        save_best=True
    )
    
    logger.info("Running final evaluation...")
    
    final_profiler = OptimizationProfiler(os.path.join(args.output_dir, 'profiles'))
    
    with tqdm(total=2, desc="Benchmarking", leave=False) as pbar:
        teacher_final = final_profiler.profile_model(teacher, device=device)
        pbar.update(1)
        student_final = final_profiler.profile_model(student, device=device)
        pbar.update(1)
    
    speedup = teacher_final['latency']['mean'] / student_final['latency']['mean']
    
    logger.info("=" * 60)
    logger.info("FINAL OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Teacher latency: {teacher_final['latency']['mean']:.2f}±{teacher_final['latency']['std']:.2f} ms")
    logger.info(f"Student latency: {student_final['latency']['mean']:.2f}±{student_final['latency']['std']:.2f} ms")
    logger.info(f"Speedup: {speedup:.1f}x")
    logger.info(f"Compression: {teacher_params/student_params:.1f}x")
    logger.info(f"Memory reduction: {teacher_final['memory_mb']/max(student_final['memory_mb'], 1):.1f}x")
    logger.info("=" * 60)
    
    if QUANT_AVAILABLE:
        logger.info("Applying INT8 quantization to student model...")
        quantized_student = quantize_model(student, val_loader, device, num_calibration_batches=5)
        
        logger.info("Profiling quantized model...")
        quantized_final = final_profiler.profile_model(quantized_student, device=device)
        
        quantized_speedup = teacher_final['latency']['mean'] / quantized_final['latency']['mean']
        quantized_compression = teacher_params / sum(p.numel() for p in quantized_student.parameters())
        
        logger.info(f"Quantized model latency: {quantized_final['latency']['mean']:.2f}±{quantized_final['latency']['std']:.2f} ms")
        logger.info(f"Quantized model speedup: {quantized_speedup:.1f}x")
        logger.info(f"Quantized model compression: {quantized_compression:.1f}x")
    else:
        logger.info("Quantization not available in this PyTorch version, skipping")
    
    logger.info("Applying structured pruning to student model...")
    pruned_student = apply_structured_pruning(student, sparsity=0.25)  # 25% sparsity
    
    logger.info("Profiling pruned model...")
    pruned_final = final_profiler.profile_model(pruned_student, device=device)
    
    pruned_speedup = teacher_final['latency']['mean'] / pruned_final['latency']['mean']
    pruned_compression = teacher_params / sum(p.numel() for p in pruned_student.parameters())
    
    logger.info(f"Pruned model latency: {pruned_final['latency']['mean']:.2f}±{pruned_final['latency']['std']:.2f} ms")
    logger.info(f"Pruned model speedup: {pruned_speedup:.1f}x")
    logger.info(f"Pruned model compression: {pruned_compression:.1f}x")
    
    logger.info("Applying attention head pruning to student model...")
    attn_pruned_student = apply_attention_head_pruning(student, sparsity=0.3)

    logger.info("Profiling attention-pruned model...")
    attn_pruned_final = final_profiler.profile_model(attn_pruned_student, device=device)
    
    attn_pruned_speedup = teacher_final['latency']['mean'] / attn_pruned_final['latency']['mean']
    attn_pruned_compression = teacher_params / sum(p.numel() for p in attn_pruned_student.parameters())
    
    logger.info(f"Attention-pruned model latency: {attn_pruned_final['latency']['mean']:.2f}±{attn_pruned_final['latency']['std']:.2f} ms")
    logger.info(f"Attention-pruned model speedup: {attn_pruned_speedup:.1f}x")
    logger.info(f"Attention-pruned model compression: {attn_pruned_compression:.1f}x")
    
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
