import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
import time
import os
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


class OptimizationProfiler:
    
    def __init__(self, output_dir="profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {}
    
    def profile_model(self, model, input_shape=(1, 2048), device='cpu'):
        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                _ = model(dummy_input)
                latencies.append((time.perf_counter() - start) * 1000)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(dummy_input)
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory = 0
        
        self.metrics['latency'] = {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies))
        }
        self.metrics['memory_mb'] = peak_memory
        self.metrics['parameters'] = model.get_num_parameters() if hasattr(model, 'get_num_parameters') else sum(p.numel() for p in model.parameters())
        
        return self.metrics


class DistillationTrainer:
    
    def __init__(self, 
                 teacher_model, 
                 student_model, 
                 temperature=3.0, 
                 alpha=0.7,
                 device=None):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.teacher.to(self.device)
        self.student.to(self.device)

        self.teacher.eval()
        self.profiler = OptimizationProfiler()
    
    def soft_cross_entropy(self, logits, targets):
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        targets_prob = F.softmax(targets / self.temperature, dim=-1)
        return -(targets_prob * log_probs).sum(dim=-1).mean()
    
    def train_epoch(self, 
                    train_loader, 
                    optimizer, 
                    use_amp=False, 
                    scaler=None,
                    scheduler=None,
                    grad_accum_steps=1):
        self.student.train()
        total_loss = 0

        batch_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (img, state, action) in enumerate(batch_pbar):
            img, state, action = img.to(self.device), state.to(self.device), action.to(self.device)
            
            with torch.no_grad():
                teacher_out = self.teacher(img, state)

                if teacher_out.shape[-1] != 14:
                    if teacher_out.shape[-1] < 14:
                        pad_size = 14 - teacher_out.shape[-1]
                        teacher_out = torch.cat([
                            teacher_out,
                            torch.zeros(teacher_out.shape[0], pad_size, device=teacher_out.device)
                        ], dim=-1)
                    else:

                        teacher_out = teacher_out[:, :14]
            
            if use_amp:
                with autocast():
                    student_out = self.student(img, state)

                    if student_out.shape[-1] != teacher_out.shape[-1]:
                        if student_out.shape[-1] < teacher_out.shape[-1]:
                            # Pad student output
                            pad_size = teacher_out.shape[-1] - student_out.shape[-1]
                            student_out = torch.cat([
                                student_out,
                                torch.zeros(student_out.shape[0], pad_size, device=student_out.device)
                            ], dim=-1)
                        else:
                            student_out = student_out[:, :teacher_out.shape[-1]]

                    soft_loss = self.soft_cross_entropy(student_out, teacher_out)
                    hard_loss = F.mse_loss(student_out, teacher_out)
                    loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

                    loss = loss / grad_accum_steps
                
                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() 
            else:
                student_out = self.student(img, state)

                if student_out.shape[-1] != teacher_out.shape[-1]:
                    if student_out.shape[-1] < teacher_out.shape[-1]:
                        pad_size = teacher_out.shape[-1] - student_out.shape[-1]
                        student_out = torch.cat([
                            student_out,
                            torch.zeros(student_out.shape[0], pad_size, device=student_out.device)
                        ], dim=-1)
                    else:
                        student_out = student_out[:, :teacher_out.shape[-1]]

                soft_loss = self.soft_cross_entropy(student_out, teacher_out)
                hard_loss = F.mse_loss(student_out, teacher_out)
                loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

                loss = loss / grad_accum_steps
                
                loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()  # Reset gradients after step
            
            total_loss += loss.item() * grad_accum_steps  # Multiply back to get actual loss

            batch_pbar.set_postfix({
                'loss': f'{loss.item() * grad_accum_steps:.4f}',  # Show actual loss
                'lr': f'{scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the student model."""
        self.student.eval()
        val_loss = 0
        val_samples = 0
        
        with tqdm(val_loader, desc="Validation", leave=False) as val_pbar:
            for img, state, action in val_pbar:
                img, state, action = img.to(self.device), state.to(self.device), action.to(self.device)
                out = self.student(img, state)

                if len(out.shape) > len(action.shape) and out.shape[-1] == action.shape[-1]:

                    out = out[..., -action.shape[-1]:]
                elif out.shape[-1] > action.shape[-1]:

                    out = out[..., :action.shape[-1]]
                elif out.shape[-1] < action.shape[-1]:

                    pad_size = action.shape[-1] - out.shape[-1]
                    out = torch.cat([out, torch.zeros(*out.shape[:-1], pad_size, device=out.device)], dim=-1)

                if out.shape != action.shape:
                    min_len = min(out.shape[-1], action.shape[-1])
                    out = out[..., :min_len]
                    action = action[..., :min_len]
                
                val_loss += F.mse_loss(out, action).item() * img.size(0)
                val_samples += img.size(0)

                val_pbar.set_postfix({'val_loss': f'{F.mse_loss(out, action).item():.4f}'})
        
        return val_loss / max(val_samples, 1)
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs=10, 
              lr=1e-4, 
              mixed_precision=False,
              output_dir='results',
              save_best=True,
              early_stopping_patience=None,
              early_stopping_min_delta=1e-4,
              grad_accum_steps=1):

        optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        use_amp = mixed_precision and hasattr(torch.cuda, 'amp') and self.device.type == 'cuda'
        scaler = GradScaler() if use_amp else None

        if early_stopping_patience is not None:
            self.enable_early_stopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
        
        if use_amp:
            logger.info("Mixed precision training enabled")

        best_loss = float('inf')
        logger.info("Starting knowledge distillation training...")
        
        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
        for epoch in epoch_pbar:
            avg_train_loss = self.train_epoch(train_loader, optimizer, use_amp, scaler, scheduler, grad_accum_steps=grad_accum_steps)
            
            avg_val_loss = self.validate(val_loader)
            
            if early_stopping_patience is not None:
                if self.check_early_stopping(avg_val_loss):
                    logger.info(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                    break
            
            scheduler.step()

            epoch_pbar.set_postfix({
                'train': f'{avg_train_loss:.4f}',
                'val': f'{avg_val_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            logger.info(f"Epoch {epoch:2d}/{epochs} | "
                       f"Train: {avg_train_loss:.4f} | "
                       f"Val: {avg_val_loss:.4f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.2e}")
            if save_best and avg_val_loss < best_loss:
                best_loss = avg_val_loss
                os.makedirs(output_dir, exist_ok=True)
                torch.save({
                    'student_state_dict': self.student.state_dict(),
                    'teacher_state_dict': self.teacher.state_dict() if hasattr(self.teacher, 'state_dict') else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                }, os.path.join(output_dir, 'best_model.pth'))

            if epoch % 5 == 0:  # Save checkpoint every 5 epochs
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'student_state_dict': self.student.state_dict(),
                    'teacher_state_dict': self.teacher.state_dict() if hasattr(self.teacher, 'state_dict') else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'best_loss': best_loss
                }, checkpoint_path)
        
        logger.info("Knowledge distillation training completed!")
        return best_loss

    def enable_early_stopping(self, patience: int = 5, min_delta: float = 1e-4):
        self.early_stopping_patience = patience
        self.early_stopping_min_delta = min_delta
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
    def check_early_stopping(self, current_val_loss: float) -> bool:
        if current_val_loss < self.best_val_loss - self.early_stopping_min_delta:
            self.best_val_loss = current_val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            
        return self.early_stopping_counter >= self.early_stopping_patience
    


def apply_structured_pruning(model, sparsity=0.2):
    print(f"Applying {sparsity*100:.0f}% structured pruning...")
    total_params_before = sum(p.numel() for p in model.parameters())
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            try:
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
            except Exception as e:
                print(f"Could not prune layer {name}: {e}")
                continue
    
    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"Pruning completed: {total_params_before} -> {total_params_after} parameters ({(1-total_params_after/total_params_before)*100:.1f}% reduction)")
    return model


def apply_attention_head_pruning(model, sparsity=0.3):
    print(f"Applying {sparsity*100:.0f}% pruning to attention heads...")
    
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            if isinstance(module, nn.Linear):
                try:
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')
                    print(f"  Pruned attention layer: {name}")
                except Exception as e:
                    print(f"  Could not prune attention layer {name}: {e}")
    
    print("Attention head pruning completed")
    return model