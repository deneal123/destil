import torch
import torch.nn as nn

from .constants import (
    STUDENT_INPUT_DIM,
    STUDENT_MIN_HIDDEN_DIM,
    STUDENT_ACTION_DIM,
    STUDENT_MIN_LAYERS,
    STUDENT_MIN_HEADS,
    TEACHER_ENCODER_OUTPUT_DIM,
    TEACHER_TRANSFORMER_NUM_LAYERS,
    TEACHER_TRANSFORMER_NHEAD
)


class StudentModel(nn.Module):
    
    def __init__(self, ratio=0.5):
        super().__init__()
        hidden = max(int(TEACHER_ENCODER_OUTPUT_DIM * ratio), STUDENT_MIN_HIDDEN_DIM)
        self.encoder = nn.Sequential(
            nn.Linear(STUDENT_INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        layers = max(int(TEACHER_TRANSFORMER_NUM_LAYERS * ratio), STUDENT_MIN_LAYERS)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, 
                nhead=max(int(TEACHER_TRANSFORMER_NHEAD * ratio), STUDENT_MIN_HEADS), 
                dim_feedforward=hidden*2, 
                batch_first=True
            ),
            num_layers=layers
        )
        self.action_head = nn.Linear(hidden, STUDENT_ACTION_DIM)
    
    def forward(self, img_features, state=None):
        x = self.encoder(img_features)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.action_head(x)
    
    def get_features(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        return x.squeeze(1)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    


class QuantizedSmolVLAModel(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, img_features, state=None, **kwargs):
        img_features = self.quant(img_features)
        if hasattr(self.model, 'forward'):
            output = self.model.forward(img_features, state, **kwargs)
        else:
            output = self.model(img_features, state, **kwargs)
        output = self.dequant(output)
        return output
    
    def prepare_qat(self):
        return torch.quantization.prepare_qat(self, inplace=False)
    
    def convert(self):
        return torch.quantization.convert(self, inplace=False)


def quantize_model(model, dataloader, device, num_calibration_batches=10):
    quantized_model = QuantizedSmolVLAModel(model)

    quantized_model.eval()
    quantized_model.model.eval()
    
    print(f"Starting quantization calibration with {num_calibration_batches} batches...")
    with torch.no_grad():
        calib_count = 0
        for batch_idx, (img, state, action) in enumerate(dataloader):
            if calib_count >= num_calibration_batches:
                break
            
            img, state = img.to(device), state.to(device)
            
            _ = quantized_model(img, state)
            calib_count += 1
    
    quantized_model_converted = quantized_model.convert()
    
    print("Quantization completed!")
    return quantized_model_converted