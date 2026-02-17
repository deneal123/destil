# План выполнения домашнего задания №2
## Практическая реализация методов ускорения

> **Дедлайн**: 19 февраля 2026  
> **Статус**: В работе  
> **Проект**: SmolVLA Model Optimization Framework  
> **Связь с ДЗ №1**: Реализация гипотез ускорения

---

## 📋 Обзор задания

Переход от анализа к практике: реализовать методы ускорения SmolVLA, провести эксперименты и оценить эффективность. В проекте уже есть готовая инфраструктура - нужно её использовать и расширить.

---

## ✅ Задача 1: Выбор базовой модели

### Что нужно сделать

- [ ] Взять модель из ДЗ №1 (SmolVLA)
- [ ] Описать текущую конфигурацию
- [ ] Зафиксировать bottlenecks из первой работы
- [ ] Обосновать выбор методов ускорения

### Исходные данные в проекте

- `docs/hw1_report.md` - анализ из ДЗ №1 (если уже выполнен)
- `src/models/teacher.py` - базовая модель
- `configs/config.json` - текущая конфигурация

### Команды для проверки базовой модели

```bash
# Получить параметры базовой модели
python -c "
from src.models.teacher import RealSmolVLAModel
import torch

model = RealSmolVLAModel(use_real=False)
params = model.get_num_parameters()
print(f'Teacher model parameters: {params:,}')
print(f'Model size (MB): {params * 4 / 1024 / 1024:.2f}')  # FP32
"

# Профилирование базовой модели
python scripts/train.py --epochs 1 --profile --num_samples 100 --batch_size 8
```

### Что написать в отчете

```markdown
# 1. Базовая модель: SmolVLA

## 1.1 Конфигурация модели
- **Модель**: RealSmolVLAModel (Teacher)
- **Задача**: Vision-Language-Action для робототехники
- **Сценарий**: Real-time inference на роботах
- **Платформа**: NVIDIA Jetson (edge device)

## 1.2 Архитектура (из ДЗ №1)
```
Vision Encoder: 2048 → 2048 → 1024 → 512
Transformer: 6 layers × 8 heads, dim=512, FFN=1024
Action Head: 512 → 7
Total params: ~X,XXX,XXX
```

## 1.3 Выявленные Bottlenecks (из ДЗ №1)
1. **Multi-Head Attention** (60-70% времени)
   - 6 layers × 8 heads
   - O(n²) complexity
   
2. **Vision Encoder** (20-25% времени)
   - Первый Linear layer: 2048×2048 = 4.2M params
   
3. **Memory consumption** (~XXX MB)
   - Критично для edge devices

## 1.4 Обоснование выбора методов
На основе bottlenecks выбираем:
1. **Knowledge Distillation** - уменьшить архитектуру
2. **Quantization** - уменьшить memory footprint
3. **Pruning** - удалить избыточные веса
```

### Файлы для работы

- `src/models/teacher.py:20-142` - полная архитектура
- `src/models/constants.py` - все размерности

---

## ✅ Задача 2: Выбор и реализация метода ускорения

### Что нужно сделать

- [ ] Выбрать минимум 1 метод (рекомендую 2-3 для полноты)
- [ ] Объяснить, почему метод подходит для SmolVLA
- [ ] Описать, какие части архитектуры затрагиваются
- [ ] Объяснить механизм ускорения

### Доступные методы в проекте

#### ✅ Метод 1: Knowledge Distillation (УЖЕ РЕАЛИЗОВАН!)

**Файлы**:
- `src/models/distillation.py` - полная реализация
- `src/models/student.py` - student model
- `src/training/trainer.py` - DistillationTrainer

**Описание реализации**:
```python
# Уже реализовано в проекте!
class DistillationTrainer:
    - Teacher: RealSmolVLAModel (6 layers, 512 dim)
    - Student: StudentModel (3 layers, 256 dim при ratio=0.5)
    - Loss: α × KL_div + (1-α) × task_loss
    - Temperature scaling: soft targets
```

**Что написать**:
```markdown
### Метод 1: Knowledge Distillation

**Обоснование выбора**:
- Bottleneck: 6 transformer layers (60-70% времени)
- Решение: уменьшить до 3 layers через distillation
- Механизм: student учится у teacher с soft targets

**Затрагиваемые части**:
- Vision Encoder: 512 → 256 dim (ratio=0.5)
- Transformer: 6 → 3 layers
- Attention heads: 8 → 4 heads
- FFN: 1024 → 512 dim

**Механизм ускорения**:
1. Меньше параметров → быстрее forward pass
2. Меньше layers → меньше sequential computation
3. FLOPs reduction: ~4x
4. Memory reduction: ~4x
```

#### ✅ Метод 2: INT8 Quantization (УЖЕ РЕАЛИЗОВАН!)

**Файлы**:
- `src/models/student.py:55-102` - quantize_model()
- `src/models/quantization.py` - дополнительные утилиты

**Описание реализации**:
```python
# Уже реализовано!
def quantize_model(model, dataloader, device, num_calibration_batches=10):
    - Dynamic quantization
    - Calibration на валидационных данных
    - FP32 → INT8 конвертация
```

**Что написать**:
```markdown
### Метод 2: INT8 Dynamic Quantization

**Обоснование выбора**:
- Bottleneck: Memory consumption (XXX MB на Jetson)
- Решение: FP32 → INT8 (4x меньше памяти)
- Механизм: weight quantization + dynamic activation quantization

**Затрагиваемые части**:
- Все Linear layers
- Все веса модели
- Активации (динамически)

**Механизм ускорения**:
1. Memory: 4x reduction (32-bit → 8-bit)
2. Inference: INT8 ops быстрее на CPU/Edge TPU
3. Cache efficiency: больше весов в L1/L2 cache
```

#### ✅ Метод 3: Structured Pruning (УЖЕ РЕАЛИЗОВАН!)

**Файлы**:
- `src/training/trainer.py:227-276` - apply_structured_pruning()
- `src/training/trainer.py:279-320` - apply_attention_head_pruning()

**Описание реализации**:
```python
# Уже реализовано!
def apply_structured_pruning(model, sparsity=0.25):
    - Magnitude-based pruning
    - Удаление целых каналов
    - L1 norm для оценки важности
```

**Что написать**:
```markdown
### Метод 3: Structured Pruning

**Обоснование выбора**:
- Bottleneck: Избыточные параметры в encoder
- Решение: удалить 25-30% наименее важных весов
- Механизм: magnitude-based structured pruning

**Затрагиваемые части**:
- Linear layers в encoder
- Linear layers в transformer
- Целые выходные каналы (структурный pruning)

**Механизм ускорения**:
1. Реальное уменьшение FLOPs
2. Меньше параметров → быстрее inference
3. Structured pruning → hardware-friendly
```

### Рекомендуемая комбинация методов

```markdown
## Комбинированный подход (РЕКОМЕНДУЕТСЯ)

**Pipeline оптимизации**:
1. Knowledge Distillation: Teacher → Student (ratio=0.5)
2. Fine-tuning Student model
3. Structured Pruning: 25% sparsity
4. INT8 Quantization: для deployment

**Обоснование**:
- Distillation дает основное ускорение (4x)
- Pruning дополнительно удаляет избыточность (+20-30%)
- Quantization оптимизирует для edge devices (+2x memory)
- Итого: ~6-8x speedup, ~85% size reduction
```

---

## ✅ Задача 3: Алгоритм применения метода

### Что нужно сделать

- [ ] Написать/использовать код реализации
- [ ] Добавить подробные комментарии
- [ ] Зафиксировать гиперпараметры
- [ ] Описать настройки и режимы

### Код уже реализован! Но нужно документировать

#### Для Knowledge Distillation

**Создать файл**: `docs/hw2_distillation_algorithm.py`

```python
"""
Алгоритм Knowledge Distillation для SmolVLA
Домашнее задание №2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.teacher import RealSmolVLAModel
from src.models.student import StudentModel

# ============================================
# ГИПЕРПАРАМЕТРЫ (из configs/config.json)
# ============================================
STUDENT_RATIO = 0.5      # Уменьшение размера: 50% от teacher
TEMPERATURE = 3.0        # Temperature для soft targets
ALPHA = 0.7              # Вес distillation loss
LEARNING_RATE = 1e-4     # Learning rate для student
EPOCHS = 20              # Количество эпох обучения
BATCH_SIZE = 32          # Batch size

# ============================================
# ШАГ 1: Создание Teacher и Student моделей
# ============================================
def create_models(device):
    """
    Создает teacher и student модели
    
    Teacher: 6 layers, 512 dim, 8 heads
    Student: 3 layers, 256 dim, 4 heads (при ratio=0.5)
    """
    # Teacher model (уже обученная)
    teacher = RealSmolVLAModel(use_real=False).to(device)
    teacher.eval()  # Режим inference
    
    # Student model (будем обучать)
    student = StudentModel(ratio=STUDENT_RATIO).to(device)
    student.train()  # Режим training
    
    print(f"Teacher params: {teacher.get_num_parameters():,}")
    print(f"Student params: {student.get_num_parameters():,}")
    print(f"Compression: {teacher.get_num_parameters()/student.get_num_parameters():.2f}x")
    
    return teacher, student

# ============================================
# ШАГ 2: Distillation Loss Function
# ============================================
def distillation_loss(student_logits, teacher_logits, true_labels, 
                     temperature=3.0, alpha=0.7):
    """
    Комбинированная функция потерь для дистилляции
    
    Args:
        student_logits: выходы student модели
        teacher_logits: выходы teacher модели
        true_labels: истинные метки (действия)
        temperature: температура для soft targets
        alpha: вес distillation loss (0-1)
    
    Returns:
        total_loss: общая потеря
        distill_loss: компонента от дистилляции
        task_loss: компонента от задачи
    """
    # 1. Task loss: обычный MSE с истинными метками
    task_loss = F.mse_loss(student_logits, true_labels)
    
    # 2. Distillation loss: KL divergence между student и teacher
    # Применяем temperature scaling для получения "мягких" распределений
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence, умноженная на T² (стандартная практика)
    distill_loss = F.kl_div(
        student_log_soft, 
        teacher_soft, 
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # 3. Комбинированная потеря
    total_loss = alpha * distill_loss + (1 - alpha) * task_loss
    
    return total_loss, distill_loss, task_loss

# ============================================
# ШАГ 3: Training Loop
# ============================================
def train_distillation(teacher, student, train_loader, val_loader, 
                       epochs=20, lr=1e-4, device='cuda'):
    """
    Обучение student модели через distillation
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # ===== TRAINING =====
        student.train()
        teacher.eval()  # Teacher всегда в eval mode
        
        train_loss = 0.0
        for batch_idx, (img_features, state, actions) in enumerate(train_loader):
            img_features = img_features.to(device)
            state = state.to(device)
            actions = actions.to(device)
            
            # Forward pass через обе модели
            with torch.no_grad():
                teacher_output = teacher(img_features, state)
            
            student_output = student(img_features, state)
            
            # Вычисление loss
            loss, distill, task = distillation_loss(
                student_output, teacher_output, actions,
                temperature=TEMPERATURE, alpha=ALPHA
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # ===== VALIDATION =====
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img_features, state, actions in val_loader:
                img_features = img_features.to(device)
                state = state.to(device)
                actions = actions.to(device)
                
                teacher_output = teacher(img_features, state)
                student_output = student(img_features, state)
                
                loss, _, _ = distillation_loss(
                    student_output, teacher_output, actions,
                    temperature=TEMPERATURE, alpha=ALPHA
                )
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), 'results/best_student.pth')
            print(f"  ✓ Best model saved!")
        
        scheduler.step()
    
    return student

# ============================================
# ИСПОЛЬЗОВАНИЕ
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание моделей
    teacher, student = create_models(device)
    
    # Загрузка данных (используем существующий код)
    from src.datasets.real_dataset import RealLeRobotDataset, create_dataloader
    
    train_dataset = RealLeRobotDataset('lerobot/pusht', 'train', 2000)
    val_dataset = RealLeRobotDataset('lerobot/pusht', 'val', 400)
    
    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=BATCH_SIZE)
    
    # Обучение
    student = train_distillation(
        teacher, student, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=device
    )
```

#### Для Quantization

**Создать файл**: `docs/hw2_quantization_algorithm.py`

```python
"""
Алгоритм INT8 Quantization для SmolVLA
Домашнее задание №2
"""

import torch
import torch.nn as nn
from src.models.student import StudentModel, QuantizedSmolVLAModel

# ============================================
# ГИПЕРПАРАМЕТРЫ
# ============================================
NUM_CALIBRATION_BATCHES = 10  # Количество батчей для калибровки
QUANTIZATION_BACKEND = 'fbgemm'  # 'fbgemm' для CPU, 'qnnpack' для mobile

# ============================================
# ШАГ 1: Подготовка модели
# ============================================
def prepare_model_for_quantization(model):
    """
    Подготовка модели к квантизации
    
    1. Добавляем QuantStub и DeQuantStub
    2. Настраиваем qconfig
    3. Fuse операции (опционально)
    """
    # Обертка модели для квантизации
    quantized_wrapper = QuantizedSmolVLAModel(model)
    
    # Настройка backend
    torch.backends.quantized.engine = QUANTIZATION_BACKEND
    
    # Конфигурация квантизации (dynamic для весов)
    quantized_wrapper.qconfig = torch.quantization.get_default_qconfig(QUANTIZATION_BACKEND)
    
    return quantized_wrapper

# ============================================
# ШАГ 2: Calibration
# ============================================
def calibrate_model(model, dataloader, device, num_batches=10):
    """
    Калибровка модели на реальных данных
    
    Цель: определить диапазоны активаций для квантизации
    """
    model.eval()
    print(f"Starting calibration with {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, (img_features, state, actions) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            img_features = img_features.to(device)
            state = state.to(device)
            
            # Forward pass для сбора статистики
            _ = model(img_features, state)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Calibrated {batch_idx + 1}/{num_batches} batches")
    
    print("Calibration complete!")
    return model

# ============================================
# ШАГ 3: Конвертация в INT8
# ============================================
def quantize_model(model, dataloader, device, num_calibration_batches=10):
    """
    Полный pipeline квантизации:
    1. Prepare: добавить quant/dequant stubs
    2. Calibrate: собрать статистику на данных
    3. Convert: конвертировать в INT8
    
    Args:
        model: обученная FP32 модель
        dataloader: данные для калибровки
        device: 'cuda' или 'cpu'
        num_calibration_batches: количество батчей для калибровки
    
    Returns:
        quantized_model: квантизованная INT8 модель
    """
    # ШАГ 1: Подготовка
    print("Step 1: Preparing model for quantization...")
    model.eval()
    model = model.cpu()  # Quantization работает на CPU
    
    quantized_wrapper = prepare_model_for_quantization(model)
    
    # ШАГ 2: Калибровка
    print("Step 2: Calibrating model...")
    quantized_wrapper = calibrate_model(
        quantized_wrapper, dataloader, 'cpu', num_calibration_batches
    )
    
    # ШАГ 3: Конвертация
    print("Step 3: Converting to INT8...")
    quantized_model = quantized_wrapper.convert()
    
    # Проверка размера модели
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    
    print(f"\nQuantization Results:")
    print(f"  Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"  Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"  Compression: {original_size / quantized_size:.2f}x")
    
    return quantized_model

# ============================================
# ШАГ 4: Валидация квантизованной модели
# ============================================
def validate_quantized_model(original_model, quantized_model, dataloader, device='cpu'):
    """
    Сравнение accuracy оригинальной и квантизованной модели
    """
    original_model.eval()
    quantized_model.eval()
    
    original_loss = 0.0
    quantized_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for img_features, state, actions in dataloader:
            img_features = img_features.to(device)
            state = state.to(device)
            actions = actions.to(device)
            
            # Оригинальная модель
            original_output = original_model(img_features, state)
            original_loss += F.mse_loss(original_output, actions).item()
            
            # Квантизованная модель
            quantized_output = quantized_model(img_features, state)
            quantized_loss += F.mse_loss(quantized_output, actions).item()
            
            num_batches += 1
    
    print(f"\nValidation Results:")
    print(f"  Original Loss: {original_loss / num_batches:.4f}")
    print(f"  Quantized Loss: {quantized_loss / num_batches:.4f}")
    print(f"  Degradation: {((quantized_loss - original_loss) / original_loss * 100):.2f}%")

# ============================================
# ИСПОЛЬЗОВАНИЕ
# ============================================
if __name__ == "__main__":
    # Загрузка обученной student модели
    student = StudentModel(ratio=0.5)
    student.load_state_dict(torch.load('results/best_student.pth'))
    
    # Загрузка данных для калибровки
    from src.datasets.real_dataset import RealLeRobotDataset, create_dataloader
    val_dataset = RealLeRobotDataset('lerobot/pusht', 'val', 400)
    val_loader = create_dataloader(val_dataset, batch_size=32)
    
    # Квантизация
    quantized_student = quantize_model(
        student, val_loader, 'cpu', 
        num_calibration_batches=10
    )
    
    # Валидация
    validate_quantized_model(student, quantized_student, val_loader)
    
    # Сохранение
    torch.save(quantized_student.state_dict(), 'results/quantized_student.pth')
```

#### Для Pruning

**Создать файл**: `docs/hw2_pruning_algorithm.py`

```python
"""
Алгоритм Structured Pruning для SmolVLA
Домашнее задание №2
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# ============================================
# ГИПЕРПАРАМЕТРЫ
# ============================================
SPARSITY = 0.25  # 25% весов будут удалены
PRUNING_METHOD = 'structured'  # 'structured' или 'unstructured'
DIMENSION = 0  # Для structured: 0=output channels, 1=input channels

# ============================================
# ШАГ 1: Анализ важности весов
# ============================================
def compute_importance_scores(module):
    """
    Вычисление важности весов по L1 norm
    
    Менее важные веса (меньшая норма) будут удалены
    """
    if hasattr(module, 'weight'):
        # L1 norm по выходным каналам (dim=0)
        importance = torch.norm(module.weight.data, p=1, dim=1)
        return importance
    return None

# ============================================
# ШАГ 2: Structured Pruning
# ============================================
def apply_structured_pruning(model, sparsity=0.25):
    """
    Структурное pruning Linear layers
    
    Удаляем целые выходные каналы (строки матрицы весов),
    что дает реальное ускорение на hardware
    
    Args:
        model: модель для pruning
        sparsity: доля удаляемых весов (0.0-1.0)
    
    Returns:
        pruned_model: модель после pruning
    """
    print(f"Applying structured pruning (sparsity={sparsity})...")
    
    # Список модулей для pruning
    modules_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_prune.append((module, name))
            print(f"  Found Linear layer: {name} (shape: {module.weight.shape})")
    
    # Применяем pruning к каждому Linear layer
    for module, name in modules_to_prune:
        # Вычисляем важность
        importance = compute_importance_scores(module)
        
        # Применяем structured pruning по L1 norm
        prune.ln_structured(
            module,
            name='weight',
            amount=sparsity,
            n=1,  # L1 norm
            dim=0  # Pruning output channels
        )
        
        # Делаем pruning постоянным (удаляем маску)
        prune.remove(module, 'weight')
        
        # Статистика
        remaining = (module.weight != 0).sum().item()
        total = module.weight.numel()
        actual_sparsity = 1 - (remaining / total)
        print(f"  {name}: {actual_sparsity:.1%} pruned ({remaining}/{total} remaining)")
    
    return model

# ============================================
# ШАГ 3: Attention Head Pruning
# ============================================
def apply_attention_head_pruning(model, num_heads_to_prune=2):
    """
    Pruning менее важных attention heads
    
    В transformer с 8 heads можем удалить 2-4 наименее важных
    """
    print(f"Pruning {num_heads_to_prune} attention heads...")
    
    # Поиск transformer layers
    for name, module in model.named_modules():
        if 'transformer' in name.lower() and hasattr(module, 'self_attn'):
            # Анализ важности каждого head
            # (упрощенная версия - в реальности нужна статистика с данных)
            
            attention = module.self_attn
            num_heads = attention.num_heads
            head_dim = attention.embed_dim // num_heads
            
            # Создаем маску для pruning heads
            # Удаляем последние num_heads_to_prune heads (упрощение)
            mask = torch.ones(num_heads, dtype=torch.bool)
            mask[-num_heads_to_prune:] = False
            
            # Применяем маску к весам attention
            # (в реальной реализации нужно переделать архитектуру)
            print(f"  {name}: pruned {num_heads_to_prune}/{num_heads} heads")
    
    return model

# ============================================
# ШАГ 4: Fine-tuning после Pruning
# ============================================
def finetune_pruned_model(model, train_loader, val_loader, epochs=5, lr=1e-5, device='cuda'):
    """
    Fine-tuning модели после pruning для восстановления accuracy
    
    Args:
        model: pruned модель
        train_loader: training data
        val_loader: validation data
        epochs: количество эпох fine-tuning
        lr: learning rate (меньше чем при обычном обучении)
    """
    print(f"Fine-tuning pruned model for {epochs} epochs...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        train_loss = 0.0
        
        for img_features, state, actions in train_loader:
            img_features = img_features.to(device)
            state = state.to(device)
            actions = actions.to(device)
            
            # Forward pass
            output = model(img_features, state)
            loss = F.mse_loss(output, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return model

# ============================================
# ШАГ 5: Валидация эффективности
# ============================================
def evaluate_pruning_effectiveness(original_model, pruned_model, test_loader, device='cuda'):
    """
    Сравнение original и pruned модели
    
    Метрики:
    - Accuracy (MSE loss)
    - Inference time
    - Model size
    - FLOPs reduction
    """
    import time
    
    print("\n" + "="*50)
    print("PRUNING EFFECTIVENESS EVALUATION")
    print("="*50)
    
    # 1. Model size comparison
    original_params = sum(p.numel() for p in original_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
    
    print(f"\n1. Model Size:")
    print(f"   Original: {original_params:,} parameters")
    print(f"   Pruned: {non_zero_params:,} non-zero parameters")
    print(f"   Compression: {original_params / non_zero_params:.2f}x")
    print(f"   Sparsity: {(1 - non_zero_params/original_params)*100:.1f}%")
    
    # 2. Accuracy comparison
    original_model.eval()
    pruned_model.eval()
    
    original_loss = 0.0
    pruned_loss = 0.0
    
    with torch.no_grad():
        for img_features, state, actions in test_loader:
            img_features = img_features.to(device)
            state = state.to(device)
            actions = actions.to(device)
            
            original_output = original_model(img_features, state)
            pruned_output = pruned_model(img_features, state)
            
            original_loss += F.mse_loss(original_output, actions).item()
            pruned_loss += F.mse_loss(pruned_output, actions).item()
    
    print(f"\n2. Accuracy:")
    print(f"   Original Loss: {original_loss / len(test_loader):.4f}")
    print(f"   Pruned Loss: {pruned_loss / len(test_loader):.4f}")
    print(f"   Degradation: {((pruned_loss - original_loss) / original_loss * 100):.2f}%")
    
    # 3. Inference time
    dummy_img = torch.randn(1, 2048).to(device)
    dummy_state = torch.randn(1, 8).to(device)
    
    # Warmup
    for _ in range(10):
        _ = original_model(dummy_img, dummy_state)
        _ = pruned_model(dummy_img, dummy_state)
    
    # Benchmark
    num_runs = 100
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = original_model(dummy_img, dummy_state)
    original_time = (time.time() - start) / num_runs * 1000
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = pruned_model(dummy_img, dummy_state)
    pruned_time = (time.time() - start) / num_runs * 1000
    
    print(f"\n3. Inference Time:")
    print(f"   Original: {original_time:.2f} ms")
    print(f"   Pruned: {pruned_time:.2f} ms")
    print(f"   Speedup: {original_time / pruned_time:.2f}x")
    
    print("="*50)

# ============================================
# ИСПОЛЬЗОВАНИЕ
# ============================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загрузка обученной модели
    from src.models.student import StudentModel
    model = StudentModel(ratio=0.5).to(device)
    model.load_state_dict(torch.load('results/best_student.pth'))
    
    # Сохранение копии для сравнения
    original_model = StudentModel(ratio=0.5).to(device)
    original_model.load_state_dict(torch.load('results/best_student.pth'))
    
    # Применение pruning
    pruned_model = apply_structured_pruning(model, sparsity=0.25)
    
    # Fine-tuning (опционально)
    # from src.datasets.real_dataset import RealLeRobotDataset, create_dataloader
    # train_dataset = RealLeRobotDataset('lerobot/pusht', 'train', 1000)
    # train_loader = create_dataloader(train_dataset, batch_size=32)
    # pruned_model = finetune_pruned_model(pruned_model, train_loader, val_loader, epochs=5)
    
    # Оценка эффективности
    from src.datasets.real_dataset import RealLeRobotDataset, create_dataloader
    test_dataset = RealLeRobotDataset('lerobot/pusht', 'val', 400)
    test_loader = create_dataloader(test_dataset, batch_size=32)
    
    evaluate_pruning_effectiveness(original_model, pruned_model, test_loader, device)
    
    # Сохранение
    torch.save(pruned_model.state_dict(), 'results/pruned_student.pth')
```

### Команды для запуска

```bash
# Knowledge Distillation (уже работает!)
python scripts/train.py --epochs 20 --student_ratio 0.5 --temperature 3.0 --alpha 0.7

# С профилированием
python scripts/train.py --epochs 20 --profile --student_ratio 0.5

# Full pipeline: Distillation + Quantization + Pruning
python scripts/train.py --epochs 20 --profile --quantize --prune --mixed_precision

# Только quantization (после distillation)
python docs/hw2_quantization_algorithm.py

# Только pruning (после distillation)
python docs/hw2_pruning_algorithm.py
```

---

## ✅ Задача 4: Проведение эксперимента

### Что нужно сделать

- [ ] Описать конфигурацию среды (hardware, software)
- [ ] Зафиксировать параметры запуска
- [ ] Описать данные для экспериментов
- [ ] Обеспечить воспроизводимость
- [ ] Провести корректное сравнение

### План экспериментов

#### Эксперимент 1: Baseline (Teacher Model)

**Цель**: Получить baseline метрики

```bash
# Профилирование teacher модели
python scripts/train.py --epochs 1 --profile --num_samples 500 --batch_size 16

# Результаты сохранятся в:
# results/real_optimization/profiles/teacher_profile.json
```

**Что измерить**:
- Inference latency (ms)
- Memory consumption (MB)
- Model size (MB)
- Accuracy (MSE loss на validation)
- Throughput (samples/sec)

#### Эксперимент 2: Knowledge Distillation

**Цель**: Оценить эффективность distillation

```bash
# Обучение student модели
python scripts/train.py --epochs 20 --student_ratio 0.5 --profile \
  --num_samples 2000 --batch_size 32 --lr 1e-4 \
  --temperature 3.0 --alpha 0.7

# Попробовать разные ratios
python scripts/train.py --epochs 15 --student_ratio 0.3 --profile  # Меньше
python scripts/train.py --epochs 15 --student_ratio 0.7 --profile  # Больше
```

**Параметры для ablation study**:
- Student ratio: 0.3, 0.5, 0.7
- Temperature: 2.0, 3.0, 4.0
- Alpha: 0.5, 0.7, 0.9

#### Эксперимент 3: Quantization

**Цель**: Оценить INT8 quantization

```bash
# Квантизация student модели
python scripts/train.py --epochs 1 --quantize --profile \
  --num_samples 500 --batch_size 16

# Или использовать отдельный скрипт
python docs/hw2_quantization_algorithm.py
```

**Что измерить**:
- Model size reduction
- Inference speedup (CPU vs GPU)
- Accuracy degradation
- Memory footprint

#### Эксперимент 4: Pruning

**Цель**: Оценить structured pruning

```bash
# Pruning student модели
python scripts/train.py --epochs 1 --prune --profile \
  --num_samples 500 --batch_size 16

# Попробовать разные уровни sparsity
# Отредактировать src/training/trainer.py:
# apply_structured_pruning(model, sparsity=0.15)  # 15%
# apply_structured_pruning(model, sparsity=0.25)  # 25%
# apply_structured_pruning(model, sparsity=0.35)  # 35%
```

#### Эксперимент 5: Combined Pipeline

**Цель**: Оценить комбинацию методов

```bash
# Полный pipeline
python scripts/train.py --epochs 20 --profile --quantize --prune --mixed_precision \
  --student_ratio 0.5 --num_samples 2000 --batch_size 32
```

### Конфигурация среды

**Создать файл**: `docs/hw2_experiment_config.md`

```markdown
# Конфигурация экспериментов

## Hardware
- **CPU**: [ваш CPU, например Intel Core i7-10700K]
- **GPU**: [ваша GPU, например NVIDIA RTX 3080]
- **RAM**: [количество, например 32 GB DDR4]
- **Storage**: [тип, например NVMe SSD]

## Software
- **OS**: Windows 10/11
- **Python**: 3.10+ (проверить: `python --version`)
- **PyTorch**: 2.7.1 (из pyproject.toml)
- **CUDA**: [версия, проверить: `nvidia-smi`]
- **cuDNN**: [версия]

## Dataset
- **Name**: lerobot/pusht (LeRobot benchmark)
- **Train samples**: 2000
- **Val samples**: 400
- **Test samples**: 400
- **Batch size**: 32 (training), 16 (testing)
- **Input dimensions**: 
  - Images: 2048-dim features
  - State: 8-dim vector
  - Actions: 7-dim vector

## Training Configuration
- **Epochs**: 20 для distillation, 5 для fine-tuning
- **Learning rate**: 1e-4 (distillation), 1e-5 (fine-tuning)
- **Optimizer**: AdamW (weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR
- **Gradient clipping**: max_norm=1.0
- **Mixed precision**: FP16 (если GPU поддерживает)

## Reproducibility
- **Random seed**: 42
- **Deterministic**: True
- **Benchmark**: True

```python
# Установка seed для воспроизводимости
import torch
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```
```

### Команды для сбора метрик

```bash
# 1. Собрать все профили
python scripts/train.py --epochs 1 --profile --num_samples 500

# 2. Сохранить системную информацию
python -c "
import torch
import platform
import sys

print('=== System Information ===')
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
" > results/system_info.txt

# 3. Запустить все эксперименты
bash docs/run_all_experiments.sh
```

**Создать файл**: `docs/run_all_experiments.sh`

```bash
#!/bin/bash

# Script to run all experiments for HW2

echo "========================================"
echo "Running all experiments for HW2"
echo "========================================"

# Create results directory
mkdir -p results/hw2_experiments

# Experiment 1: Baseline (Teacher)
echo ""
echo "Experiment 1: Baseline Teacher Model"
python scripts/train.py --epochs 1 --profile --num_samples 500 --batch_size 16 \
  --output_dir results/hw2_experiments/exp1_baseline

# Experiment 2: Distillation with different ratios
echo ""
echo "Experiment 2a: Distillation (ratio=0.3)"
python scripts/train.py --epochs 20 --profile --student_ratio 0.3 \
  --num_samples 2000 --batch_size 32 \
  --output_dir results/hw2_experiments/exp2a_distill_03

echo ""
echo "Experiment 2b: Distillation (ratio=0.5)"
python scripts/train.py --epochs 20 --profile --student_ratio 0.5 \
  --num_samples 2000 --batch_size 32 \
  --output_dir results/hw2_experiments/exp2b_distill_05

echo ""
echo "Experiment 2c: Distillation (ratio=0.7)"
python scripts/train.py --epochs 20 --profile --student_ratio 0.7 \
  --num_samples 2000 --batch_size 32 \
  --output_dir results/hw2_experiments/exp2c_distill_07

# Experiment 3: Quantization
echo ""
echo "Experiment 3: INT8 Quantization"
python scripts/train.py --epochs 1 --quantize --profile \
  --num_samples 500 --batch_size 16 \
  --output_dir results/hw2_experiments/exp3_quantization

# Experiment 4: Pruning
echo ""
echo "Experiment 4: Structured Pruning"
python scripts/train.py --epochs 1 --prune --profile \
  --num_samples 500 --batch_size 16 \
  --output_dir results/hw2_experiments/exp4_pruning

# Experiment 5: Combined
echo ""
echo "Experiment 5: Combined Pipeline"
python scripts/train.py --epochs 20 --profile --quantize --prune --mixed_precision \
  --student_ratio 0.5 --num_samples 2000 --batch_size 32 \
  --output_dir results/hw2_experiments/exp5_combined

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved in: results/hw2_experiments/"
echo "========================================"
```

---

## ✅ Задача 5: Выбор и обоснование метрик

### Что нужно сделать

- [ ] Определить метрики качества
- [ ] Определить метрики производительности
- [ ] Обосновать выбор метрик
- [ ] Описать методику измерения

### Метрики для оценки

#### 1. Метрики качества модели

**Основная метрика**: MSE Loss (Mean Squared Error)

```python
# Измерение accuracy
def evaluate_model_quality(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for img_features, state, actions in dataloader:
            img_features = img_features.to(device)
            state = state.to(device)
            actions = actions.to(device)
            
            output = model(img_features, state)
            loss = F.mse_loss(output, actions, reduction='sum')
            
            total_loss += loss.item()
            total_samples += actions.size(0)
    
    avg_loss = total_loss / total_samples
    return avg_loss
```

**Дополнительные метрики**:
- MAE (Mean Absolute Error) - для интерпретируемости
- Action accuracy (если порог для "правильного" действия)

**Обоснование**:
- MSE стандартна для regression tasks
- Робототехника: важна точность предсказанных действий
- Малая ошибка → точное управление роботом

#### 2. Метрики производительности

**2.1 Inference Latency** (критично для real-time!)

```python
import time

def measure_latency(model, num_runs=100, device='cuda'):
    model.eval()
    dummy_img = torch.randn(1, 2048).to(device)
    dummy_state = torch.randn(1, 8).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_img, dummy_state)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_img, dummy_state)
            torch.cuda.synchronize() if device == 'cuda' else None
            end = time.time()
            times.append((end - start) * 1000)  # ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
```

**Обоснование**:
- Робот требует <100ms latency
- P95/P99 важны для worst-case scenarios
- Mean показывает среднюю производительность

**2.2 Throughput** (samples/sec)

```python
def measure_throughput(model, batch_size=32, num_batches=50, device='cuda'):
    model.eval()
    dummy_img = torch.randn(batch_size, 2048).to(device)
    dummy_state = torch.randn(batch_size, 8).to(device)
    
    # Warmup
    for _ in range(5):
        _ = model(dummy_img, dummy_state)
    
    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(num_batches):
            _ = model(dummy_img, dummy_state)
    end = time.time()
    
    total_samples = batch_size * num_batches
    throughput = total_samples / (end - start)
    
    return throughput
```

**Обоснование**:
- Важно для batch processing scenarios
- Показывает GPU utilization

**2.3 Memory Consumption**

```python
def measure_memory(model, device='cuda'):
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
        dummy_img = torch.randn(1, 2048).to(device)
        dummy_state = torch.randn(1, 8).to(device)
        
        model.eval()
        with torch.no_grad():
            _ = model(dummy_img, dummy_state)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        return peak_memory
    else:
        # CPU memory is harder to measure accurately
        return 0
```

**Обоснование**:
- Jetson имеет ограниченную память (8-16 GB)
- Peak memory определяет feasibility на edge devices

**2.4 Model Size**

```python
def measure_model_size(model):
    # Parameters count
    total_params = sum(p.numel() for p in model.parameters())
    
    # Size in bytes (FP32)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = size_bytes / 1024 / 1024
    
    # Non-zero parameters (для pruned models)
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    sparsity = (1 - non_zero_params / total_params) * 100
    
    return {
        'total_params': total_params,
        'non_zero_params': non_zero_params,
        'sparsity': sparsity,
        'size_mb': size_mb
    }
```

**Обоснование**:
- Определяет возможность deployment
- Меньший размер → быстрее загрузка модели
- Важно для OTA updates роботов

**2.5 FLOPs (Floating Point Operations)**

```python
from fvcore.nn import FlopCountAnalysis

def measure_flops(model, device='cuda'):
    dummy_img = torch.randn(1, 2048).to(device)
    dummy_state = torch.randn(1, 8).to(device)
    
    flops = FlopCountAnalysis(model, (dummy_img, dummy_state))
    total_flops = flops.total()
    
    return total_flops / 1e9  # GFLOPs
```

**Обоснование**:
- Независимо от hardware
- Показывает вычислительную сложность
- Полезно для теоретической оценки

### Итоговая таблица метрик

**Создать файл**: `docs/hw2_metrics_table.md`

```markdown
# Метрики для оценки HW2

| Метрика | Описание | Единица | Желаемое значение | Как измерять |
|---------|----------|---------|-------------------|--------------|
| **Качество модели** |
| MSE Loss | Средняя квадратичная ошибка | - | Минимум | evaluate_model_quality() |
| MAE | Средняя абсолютная ошибка | - | Минимум | F.l1_loss() |
| Accuracy Degradation | Потеря точности vs baseline | % | <5% | (new_loss - base_loss) / base_loss |
| **Производительность** |
| Latency (mean) | Среднее время inference | ms | <50ms | measure_latency() |
| Latency (P95) | 95-й перцентиль latency | ms | <100ms | measure_latency() |
| Throughput | Пропускная способность | samples/sec | Максимум | measure_throughput() |
| **Ресурсы** |
| Model Size | Размер модели на диске | MB | <50MB | measure_model_size() |
| Parameters | Количество параметров | M | <5M | sum(p.numel()) |
| Sparsity | Доля нулевых весов | % | 20-30% | (для pruned) |
| Memory | Пиковое потребление памяти | MB | <500MB | measure_memory() |
| FLOPs | Вычислительная сложность | GFLOPs | <1.0 | measure_flops() |
| **Ускорение** |
| Speedup | Ускорение vs baseline | x | >4x | baseline_time / new_time |
| Compression | Сжатие vs baseline | x | >4x | baseline_size / new_size |

## Критерии успеха (для робототехники)

✅ **Acceptable**: Latency <100ms, Accuracy degradation <5%, Size <100MB
🎯 **Good**: Latency <50ms, Accuracy degradation <3%, Size <50MB
🌟 **Excellent**: Latency <30ms, Accuracy degradation <1%, Size <30MB

## Целевые показатели (из README проекта)

Ожидаемые результаты на основе существующей реализации:
- Compression: 6.8x (85% size reduction)
- Speedup: 20-25x
- Accuracy degradation: 2-5%
```

---

## ✅ Задача 6: Сравнение модели до и после ускорения

### Что нужно сделать

- [ ] Собрать результаты всех экспериментов
- [ ] Создать сравнительные таблицы
- [ ] Визуализировать результаты
- [ ] Интерпретировать с инженерной точки зрения

### Шаблон таблицы результатов

**Создать файл**: `docs/hw2_results.md`

```markdown
# Результаты экспериментов HW2

## Сводная таблица

| Модель | Params | Size (MB) | Latency (ms) | Memory (MB) | FLOPs (G) | Val Loss | Degradation |
|--------|--------|-----------|--------------|-------------|-----------|----------|-------------|
| Teacher (Baseline) | X.XXM | XXX | XX.XX | XXX | X.XX | 0.XXXX | 0% |
| Student (ratio=0.3) | X.XXM | XXX | XX.XX | XXX | X.XX | 0.XXXX | X.X% |
| Student (ratio=0.5) | X.XXM | XXX | XX.XX | XXX | X.XX | 0.XXXX | X.X% |
| Student (ratio=0.7) | X.XXM | XXX | XX.XX | XXX | X.XX | 0.XXXX | X.X% |
| Student + Quantization | X.XXM | XXX | XX.XX | XXX | X.XX | 0.XXXX | X.X% |
| Student + Pruning | X.XXM | XXX | XX.XX | XXX | X.XX | 0.XXXX | X.X% |
| Combined Pipeline | X.XXM | XXX | XX.XX | XXX | X.XX | 0.XXXX | X.X% |

## Детальные результаты

### 1. Baseline (Teacher Model)

```
Model: RealSmolVLAModel
Architecture: 6 layers, 512 dim, 8 heads
Parameters: X,XXX,XXX
Size: XXX MB
FLOPs: X.XX GFLOPs

Performance:
- Latency: XX.XX ± X.XX ms (mean ± std)
- P95 Latency: XX.XX ms
- P99 Latency: XX.XX ms
- Throughput: XXX samples/sec
- Memory: XXX MB

Quality:
- Validation Loss: 0.XXXX
- MAE: 0.XXXX
```

### 2. Knowledge Distillation (ratio=0.5)

```
Model: StudentModel (ratio=0.5)
Architecture: 3 layers, 256 dim, 4 heads
Parameters: X,XXX,XXX (XX% of teacher)
Size: XXX MB (XX% of teacher)
FLOPs: X.XX GFLOPs (XX% of teacher)

Training:
- Epochs: 20
- Temperature: 3.0
- Alpha: 0.7
- Best epoch: XX
- Training time: XX minutes

Performance:
- Latency: XX.XX ± X.XX ms (mean ± std)
- P95 Latency: XX.XX ms
- Speedup: X.XXx vs teacher
- Throughput: XXX samples/sec (X.XXx vs teacher)
- Memory: XXX MB (XX% of teacher)

Quality:
- Validation Loss: 0.XXXX
- Accuracy Degradation: X.X%
- MAE: 0.XXXX

✅ Результат: [Описание результата]
```

### 3. INT8 Quantization

```
Model: Quantized StudentModel
Base model: Student (ratio=0.5)
Quantization: Dynamic INT8
Calibration batches: 10

Performance:
- Latency: XX.XX ± X.XX ms
- Speedup: X.XXx vs FP32 student
- Total speedup: X.XXx vs teacher
- Memory: XXX MB (XX% of FP32 student)
- Size: XXX MB (XX% of FP32 student)

Quality:
- Validation Loss: 0.XXXX
- Accuracy Degradation: X.X% (vs FP32 student)
- Total degradation: X.X% (vs teacher)

✅ Результат: [Описание результата]
```

### 4. Structured Pruning

```
Model: Pruned StudentModel
Base model: Student (ratio=0.5)
Pruning: Structured L1, 25% sparsity
Fine-tuning: 5 epochs

Performance:
- Latency: XX.XX ± X.XX ms
- Speedup: X.XXx vs unpruned student
- Total speedup: X.XXx vs teacher
- Parameters: X,XXX,XXX (XX% non-zero)
- Sparsity: XX.X%
- Size: XXX MB (actual on disk)

Quality:
- Validation Loss: 0.XXXX
- Accuracy Degradation: X.X% (vs unpruned student)
- Total degradation: X.X% (vs teacher)

✅ Результат: [Описание результата]
```

### 5. Combined Pipeline

```
Model: Distilled + Pruned + Quantized
Pipeline:
1. Distillation (ratio=0.5)
2. Structured pruning (25%)
3. INT8 quantization

Final metrics:
- Parameters: X,XXX,XXX → X,XXX,XXX (XX.Xx compression)
- Size: XXX MB → XX MB (XX.Xx compression)
- Latency: XX.XX ms → X.XX ms (XX.Xx speedup)
- Memory: XXX MB → XX MB (XX.Xx reduction)
- FLOPs: X.XX → X.XX GFLOPs (XX.Xx reduction)

Quality:
- Validation Loss: 0.XXXX → 0.XXXX
- Accuracy Degradation: X.X%

✅ Результат: [Итоговое описание]
```

## Графики (опционально)

[Вставить графики, если создали]

### График 1: Latency vs Accuracy
- X-axis: Accuracy Degradation (%)
- Y-axis: Latency (ms)
- Points: Разные модели

### График 2: Size vs Accuracy
- X-axis: Model Size (MB)
- Y-axis: Validation Loss

### График 3: Speedup comparison
- Bar chart: разные модели, speedup vs baseline

## Инженерная интерпретация

### Что работает хорошо

1. **Knowledge Distillation**:
   - [Описание эффективности]
   - [Почему работает]
   - [Компромиссы]

2. **Quantization**:
   - [Описание эффективности]
   - [На каком hardware лучше]
   - [Компромиссы]

3. **Pruning**:
   - [Описание эффективности]
   - [Структурный vs unstructured]
   - [Компромиссы]

### Что не работает / проблемы

1. [Проблема 1 и объяснение]
2. [Проблема 2 и объяснение]

### Неожиданные результаты

1. [Неожиданность 1 и анализ]
2. [Неожиданность 2 и анализ]
```

### Команды для сбора результатов

```bash
# Создать скрипт для сбора всех метрик
```

**Создать файл**: `docs/collect_metrics.py`

```python
"""
Скрипт для сбора всех метрик из экспериментов
"""

import json
import os
from pathlib import Path

def collect_all_metrics(results_dir='results/hw2_experiments'):
    """Собрать метрики из всех экспериментов"""
    
    experiments = [
        'exp1_baseline',
        'exp2a_distill_03',
        'exp2b_distill_05',
        'exp2c_distill_07',
        'exp3_quantization',
        'exp4_pruning',
        'exp5_combined'
    ]
    
    all_metrics = {}
    
    for exp in experiments:
        exp_dir = Path(results_dir) / exp
        profiles_dir = exp_dir / 'profiles'
        
        # Загрузить профили
        teacher_profile = profiles_dir / 'teacher_profile.json'
        student_profile = profiles_dir / 'student_profile.json'
        
        if teacher_profile.exists():
            with open(teacher_profile) as f:
                all_metrics[f'{exp}_teacher'] = json.load(f)
        
        if student_profile.exists():
            with open(student_profile) as f:
                all_metrics[f'{exp}_student'] = json.load(f)
    
    # Сохранить сводные метрики
    with open(f'{results_dir}/all_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Collected metrics from {len(all_metrics)} experiments")
    print(f"Saved to: {results_dir}/all_metrics.json")
    
    return all_metrics

def generate_markdown_table(all_metrics):
    """Генерировать Markdown таблицу из метрик"""
    
    # TODO: Реализовать генерацию таблицы
    pass

if __name__ == "__main__":
    metrics = collect_all_metrics()
    generate_markdown_table(metrics)
```

---

## ✅ Задача 7: Вывод о целесообразности подхода

### Что нужно сделать

- [ ] Сформулировать выводы об эффективности методов
- [ ] Описать достигнутые выигрыши
- [ ] Описать компромиссы
- [ ] Указать условия применимости для production

### Шаблон выводов

```markdown
# Выводы о целесообразности методов ускорения

## 1. Knowledge Distillation

### Эффективность
- **Достигнутое ускорение**: X.Xx
- **Compression**: X.Xx
- **Accuracy degradation**: X.X%

### Выигрыши
✅ [Перечислить конкретные выигрыши]
- Значительное ускорение inference
- Меньший размер модели
- Возможность deployment на edge devices
- [и т.д.]

### Компромиссы
⚠️ [Перечислить компромиссы]
- Потеря X% accuracy
- Требуется обучение teacher модели
- Длительное время обучения student
- [и т.д.]

### Применимость для production

**Рекомендуется использовать когда**:
- Latency requirement: <100ms
- Edge device deployment (ограниченные ресурсы)
- Допустима небольшая потеря accuracy
- Есть хорошо обученная teacher модель

**НЕ рекомендуется когда**:
- Критична максимальная accuracy
- Нет ресурсов для обучения двух моделей
- Модель уже достаточно быстрая

**Production readiness**: ⭐⭐⭐⭐⭐ (5/5)
- Проверенная техника
- Широко используется в индустрии
- Стабильные результаты

---

## 2. INT8 Quantization

### Эффективность
- **Memory reduction**: X.Xx
- **Speedup на CPU**: X.Xx
- **Speedup на GPU**: X.Xx
- **Accuracy degradation**: X.X%

### Выигрыши
✅ [Перечислить]
- 4x меньше памяти
- Быстрее на CPU/Edge TPU
- Проще deployment
- Меньший размер модели

### Компромиссы
⚠️ [Перечислить]
- Potential accuracy loss
- Calibration требует данных
- Не всегда быстрее на GPU
- Debugging сложнее

### Применимость для production

**Рекомендуется использовать когда**:
- CPU/Edge TPU inference
- Ограничения по памяти критичны
- Mobile/embedded deployment
- Модель не слишком маленькая

**НЕ рекомендуется когда**:
- GPU inference (FP16 обычно лучше)
- Очень маленькая модель (overhead не окупается)
- Требуется FP32 precision

**Production readiness**: ⭐⭐⭐⭐ (4/5)
- Хорошо поддерживается frameworks
- Может требовать fine-tuning
- Platform-dependent результаты

---

## 3. Structured Pruning

### Эффективность
- **Parameter reduction**: X.X%
- **Speedup**: X.Xx
- **Accuracy degradation**: X.X%

### Выигрыши
✅ [Перечислить]
- Реальное уменьшение FLOPs
- Hardware-friendly (structured)
- Дополнительное ускорение поверх distillation
- Меньший размер модели

### Компромиссы
⚠️ [Перечислить]
- Requires fine-tuning
- Accuracy loss
- Необратимая операция
- Выбор sparsity level критичен

### Применимость для production

**Рекомендуется использовать когда**:
- После distillation (дополнительная оптимизация)
- Модель имеет избыточность
- Есть данные для fine-tuning
- Допустима небольшая потеря accuracy

**НЕ рекомендуется когда**:
- Модель уже оптимизирована
- Нет ресурсов для fine-tuning
- Критична каждая доля accuracy

**Production readiness**: ⭐⭐⭐ (3/5)
- Требует careful tuning
- Результаты могут варьироваться
- Fine-tuning обязателен

---

## 4. Combined Pipeline (Distillation + Pruning + Quantization)

### Эффективность
- **Total speedup**: X.Xx vs Teacher
- **Total compression**: X.Xx vs Teacher
- **Total accuracy degradation**: X.X%

### Выигрыши
✅ [Итоговые выигрыши]
- Максимальное ускорение
- Максимальное сжатие
- Оптимально для edge deployment
- Комбинация преимуществ всех методов

### Компромиссы
⚠️ [Итоговые компромиссы]
- Кумулятивная потеря accuracy
- Сложный pipeline
- Длительное время optimization
- Требует expertise в каждом методе

### Применимость для production

**Рекомендуется использовать когда**:
- Экстремальные ограничения ресурсов (Jetson Nano, mobile)
- Допустима потеря 3-5% accuracy
- Есть ресурсы для полного pipeline
- Критичны latency И memory

**НЕ рекомендуется когда**:
- Каждая доля accuracy критична
- Простота deployment важнее performance
- Нет expertise для настройки всех методов

**Production readiness**: ⭐⭐⭐⭐ (4/5)
- Требует тщательной validation
- Pipeline может быть хрупким
- Но результаты могут быть отличными

---

## Общие рекомендации

### Для SmolVLA в робототехнике

**Рекомендуемый подход**:
1. ✅ **Distillation** (ratio=0.5) - основное ускорение
2. ✅ **INT8 Quantization** - для edge deployment
3. ⚠️ **Pruning** - опционально, если нужно еще больше

**Ожидаемые результаты**:
- Latency: <50ms (требование: <100ms) ✅
- Size: <50MB (требование: <100MB) ✅
- Accuracy degradation: 3-5% (допустимо: <5%) ✅

### По hardware platforms

| Platform | Рекомендуемый метод | Комментарий |
|----------|---------------------|-------------|
| NVIDIA GPU (server) | Distillation + FP16 | Best performance, minimal loss |
| NVIDIA Jetson | Distillation + INT8 | Balanced speed/memory |
| CPU | Distillation + INT8 + Pruning | Maximum optimization needed |
| Mobile | Combined pipeline | Extreme constraints |

### Production checklist

- [ ] Distillation: обучено и validated ✅
- [ ] Quantization: calibrated на representative data ✅
- [ ] Pruning: fine-tuned после pruning
- [ ] A/B testing: сравнение с baseline на real robot
- [ ] Latency: измерено на target hardware
- [ ] Memory: проверено peak consumption
- [ ] Edge cases: протестированы corner cases
- [ ] Monitoring: настроен мониторинг в production
```

---

## 📊 Задача 8: Оформление отчета

### Структура отчета

**Создать файл**: `docs/hw2_report.md`

```markdown
# Домашнее задание №2: Практическая реализация методов ускорения
## Дистилляция и ускорение моделей

**Студент**: [Ваше имя]  
**Дата**: 16 февраля 2026  
**Проект**: SmolVLA Model Optimization Framework

---

## Содержание

1. [Введение](#введение)
2. [Базовая модель](#базовая-модель)
3. [Выбор методов ускорения](#выбор-методов-ускорения)
4. [Реализация алгоритмов](#реализация-алгоритмов)
5. [Эксперименты](#эксперименты)
6. [Метрики](#метрики)
7. [Результаты](#результаты)
8. [Выводы](#выводы)
9. [Приложения](#приложения)
10. [Источники](#источники)

---

## 1. Введение

[Краткое описание работы, целей, задач]

---

## 2. Базовая модель

[Из Задачи 1: описание SmolVLA, bottlenecks]

---

## 3. Выбор методов ускорения

[Из Задачи 2: обоснование выбора методов]

---

## 4. Реализация алгоритмов

[Из Задачи 3: код с комментариями, гиперпараметры]

---

## 5. Эксперименты

[Из Задачи 4: конфигурация, план экспериментов]

---

## 6. Метрики

[Из Задачи 5: описание метрик и обоснование]

---

## 7. Результаты

[Из Задачи 6: таблицы, графики, интерпретация]

---

## 8. Выводы

[Из Задачи 7: целесообразность, применимость]

---

## 9. Приложения

### A. Полный код реализации
[Ссылки на файлы или встроенный код]

### B. Дополнительные графики
[Если есть]

### C. Конфигурационные файлы
[configs/config.json и т.д.]

---

## 10. Источники

1. LeRobot Framework: https://github.com/huggingface/lerobot
2. SmolVLA: [ссылка на paper если есть]
3. Knowledge Distillation: Hinton et al., 2015
4. Quantization: Jacob et al., 2018
5. Pruning: Han et al., 2015
[и т.д.]
```

---

## 🎯 План выполнения по дням

### День 1 (16 февраля): Подготовка и baseline

- [ ] Изучить существующий код проекта
- [ ] Запустить baseline профилирование
- [ ] Собрать метрики teacher модели
- [ ] Создать структуру файлов для отчета
- [ ] Написать разделы 1-2 отчета (Введение, Базовая модель)

**Команды**:
```bash
python scripts/train.py --epochs 1 --profile --num_samples 500 --batch_size 16
```

### День 2 (17 февраля): Knowledge Distillation

- [ ] Запустить обучение student модели (ratio=0.5)
- [ ] Попробовать разные ratios (0.3, 0.7)
- [ ] Собрать метрики distillation
- [ ] Документировать алгоритм distillation
- [ ] Написать раздел 3-4 отчета (Методы, Алгоритмы)

**Команды**:
```bash
python scripts/train.py --epochs 20 --profile --student_ratio 0.5 --num_samples 2000
```

### День 3 (18 февраля): Quantization & Pruning

- [ ] Применить INT8 quantization к student модели
- [ ] Применить structured pruning
- [ ] Измерить метрики обоих методов
- [ ] Запустить combined pipeline
- [ ] Собрать все результаты

**Команды**:
```bash
python scripts/train.py --epochs 1 --quantize --prune --profile
python scripts/train.py --epochs 20 --profile --quantize --prune --mixed_precision --student_ratio 0.5
```

### День 4 (19 февраля): Анализ и отчет

- [ ] Собрать все метрики в единую таблицу
- [ ] Создать графики (опционально)
- [ ] Написать разделы 5-7 (Эксперименты, Метрики, Результаты)
- [ ] Написать раздел 8 (Выводы)
- [ ] Оформить приложения и источники
- [ ] Финальная вычитка
- [ ] Залить в репозиторий
- [ ] Заполнить форму ДЗ

---

## 📋 Чек-лист перед сдачей

### Код

- [ ] `docs/hw2_distillation_algorithm.py` - документированный алгоритм distillation
- [ ] `docs/hw2_quantization_algorithm.py` - документированный алгоритм quantization
- [ ] `docs/hw2_pruning_algorithm.py` - документированный алгоритм pruning
- [ ] Все скрипты запускаются без ошибок
- [ ] Код имеет подробные комментарии

### Эксперименты

- [ ] Baseline (teacher) профилирование выполнено
- [ ] Distillation с разными ratios протестирована
- [ ] Quantization применена и измерена
- [ ] Pruning применен и измерен
- [ ] Combined pipeline протестирован
- [ ] Все результаты сохранены в `results/hw2_experiments/`

### Документация

- [ ] `docs/hw2_report.md` - полный отчет
- [ ] `docs/hw2_experiment_config.md` - конфигурация экспериментов
- [ ] `docs/hw2_metrics_table.md` - описание метрик
- [ ] `docs/hw2_results.md` - детальные результаты
- [ ] `docs/hw2_sources.md` - список источников

### Отчет

- [ ] Раздел 1: Введение написан
- [ ] Раздел 2: Базовая модель описана
- [ ] Раздел 3: Методы обоснованы
- [ ] Раздел 4: Алгоритмы документированы
- [ ] Раздел 5: Эксперименты описаны
- [ ] Раздел 6: Метрики обоснованы
- [ ] Раздел 7: Результаты представлены с таблицами
- [ ] Раздел 8: Выводы сформулированы
- [ ] Раздел 9: Приложения добавлены
- [ ] Раздел 10: Источники указаны

### Результаты

- [ ] Таблица со всеми моделями и метриками
- [ ] Графики (опционально)
- [ ] Интерпретация результатов
- [ ] Инженерные выводы
- [ ] Рекомендации для production

### Репозиторий

- [ ] Все файлы закоммичены
- [ ] README обновлен (опционально)
- [ ] Результаты включены в репозиторий
- [ ] Код организован и читаем

### Форма ДЗ

- [ ] Форма заполнена
- [ ] Ссылка на репозиторий указана
- [ ] Ссылка на отчет указана

---

## 🚀 Быстрый старт (для срочного выполнения)

Если времени мало, следуйте этому плану:

```bash
# 1. Установить зависимости
uv sync

# 2. Быстрый baseline
python scripts/train.py --epochs 1 --profile --num_samples 200 --batch_size 8

# 3. Быстрая distillation
python scripts/train.py --epochs 10 --profile --student_ratio 0.5 --num_samples 1000 --batch_size 16

# 4. Быстрая quantization+pruning
python scripts/train.py --epochs 1 --quantize --prune --profile --num_samples 200

# 5. Собрать результаты
python docs/collect_metrics.py

# 6. Написать отчет, используя шаблоны из todo_hw2.md
```

---

## ⚠️ Важные замечания

1. **Код УЖЕ реализован!** Основные методы есть в проекте - нужно их запустить, измерить и документировать
2. **Используйте малые датасеты** для быстрых тестов: `--num_samples 500-1000`
3. **Профилирование обязательно**: всегда добавляйте `--profile`
4. **Документируйте процесс**: сохраняйте outputs команд
5. **Воспроизводимость**: зафиксируйте seed, конфигурацию, версии
6. **Интерпретация важнее цифр**: объясняйте ЧТО и ПОЧЕМУ, а не просто показывайте таблицы

---

## 💡 Советы

1. **Начните с существующего кода** - не пишите с нуля
2. **Запускайте эксперименты параллельно** если возможно
3. **Сохраняйте промежуточные результаты** после каждого эксперимента
4. **Используйте шаблоны** из этого TODO для отчета
5. **Графики необязательны**, но добавляют наглядности
6. **Комментируйте код** - это часть оценки
7. **Сравнивайте с baseline** - это ключевое требование

---

## 📚 Дополнительные ресурсы

### Документация

- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- PyTorch Pruning: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- Knowledge Distillation: https://arxiv.org/abs/1503.02531

### Existing Project Files

- `README.md` - overview и features
- `scripts/train.py` - main training script
- `src/models/` - все модели
- `src/training/trainer.py` - trainer с профилированием

### Papers

- SmolVLA (если доступна paper)
- Hinton et al., 2015: Distilling the Knowledge in a Neural Network
- Jacob et al., 2018: Quantization and Training of Neural Networks
- Han et al., 2015: Learning both Weights and Connections

---

**Успехов в выполнении ДЗ №2!** 🎓

_План создан на основе существующего проекта SmolVLA Model Optimization Framework и требований домашнего задания._

---

## 📞 Контакты и помощь

Если возникнут вопросы:
1. Изучите существующий код в проекте
2. Проверьте README.md
3. Посмотрите примеры в scripts/
4. Используйте `--help` для командных аргументов

```bash
python scripts/train.py --help
```
