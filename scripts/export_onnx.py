"""Export trained models to ONNX format for optimization."""

import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path

from src.models.teacher import RealSmolVLAModel
from src.models.student import StudentModel


def export_to_onnx(model, save_path, input_shape=(1, 2048)):
    model.eval()
    
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='onnx_models')
    parser.add_argument('--ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    teacher = RealSmolVLAModel()
    student = StudentModel(ratio=args.ratio)
    
    try:
        teacher.load_state_dict(checkpoint['teacher_state_dict'])
    except KeyError:
        try:
            teacher.load_state_dict(checkpoint['teacher'])
        except KeyError:
            print("Warning: Could not load teacher model. Using randomly initialized model.")
    
    try:
        student.load_state_dict(checkpoint['student_state_dict'])
    except KeyError:
        try:
            student.load_state_dict(checkpoint['student'])
        except KeyError:
            print("Warning: Could not load student model. Using randomly initialized model.")
    
    try:
        teacher_params = teacher.get_num_parameters()
    except AttributeError:
        teacher_params = sum(p.numel() for p in teacher.parameters())
    
    try:
        student_params = student.get_num_parameters()
    except AttributeError:
        student_params = sum(p.numel() for p in student.parameters())
    
    print(f"Teacher params: {teacher_params:,}")
    print(f"Student params: {student_params:,}")
    
    teacher_path = os.path.join(args.output_dir, 'teacher.onnx')
    student_path = os.path.join(args.output_dir, 'student.onnx')
    
    export_to_onnx(teacher, teacher_path)
    export_to_onnx(student, student_path)
    
    print("ONNX export completed!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"ONNX export failed with error: {e}")
        raise
