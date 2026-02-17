import torch
import torch.nn as nn
import logging

from .constants import (
    TEACHER_ENCODER_INPUT_DIM,
    TEACHER_ENCODER_HIDDEN_1,
    TEACHER_ENCODER_HIDDEN_2,
    TEACHER_ENCODER_OUTPUT_DIM,
    TEACHER_TRANSFORMER_NHEAD,
    TEACHER_TRANSFORMER_DIM_FEEDFORWARD,
    TEACHER_TRANSFORMER_NUM_LAYERS,
    TEACHER_ACTION_DIM,
    TEACHER_DROPOUT
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


class RealSmolVLAModel(nn.Module):
    
    def __init__(self, model_id="lerobot/smolvla_base", use_real=True):
        super().__init__()
        self.model_id = model_id
        self.use_real = use_real
        self.model = None
        
        if use_real:
            try:
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                self.model = SmolVLAPolicy.from_pretrained(model_id)
                logger.info(f"Loaded real SmolVLA model: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to load real model: {e}")
                logger.info("Falling back to simulated model")
                self.use_real = False
                self._create_simulated_model()
        else:
            self._create_simulated_model()
    
    def _create_simulated_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(TEACHER_ENCODER_INPUT_DIM, TEACHER_ENCODER_HIDDEN_1),
            nn.GELU(),
            nn.Dropout(TEACHER_DROPOUT),
            nn.Linear(TEACHER_ENCODER_HIDDEN_1, TEACHER_ENCODER_HIDDEN_2),
            nn.GELU(),
            nn.Dropout(TEACHER_DROPOUT),
            nn.Linear(TEACHER_ENCODER_HIDDEN_2, TEACHER_ENCODER_OUTPUT_DIM),
            nn.LayerNorm(TEACHER_ENCODER_OUTPUT_DIM)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=TEACHER_ENCODER_OUTPUT_DIM,
                nhead=TEACHER_TRANSFORMER_NHEAD,
                dim_feedforward=TEACHER_TRANSFORMER_DIM_FEEDFORWARD,
                dropout=TEACHER_DROPOUT,
                activation='gelu',
                batch_first=True
            ),
            num_layers=TEACHER_TRANSFORMER_NUM_LAYERS
        )
        self.action_head = nn.Linear(TEACHER_ENCODER_OUTPUT_DIM, TEACHER_ACTION_DIM)
    
    def forward(self, img_features, state=None, **kwargs):
        if self.use_real and self.model is not None:
            batch_size = img_features.shape[0]
            feature_dim = img_features.shape[-1]
            if feature_dim == 2048:
                h, w = 32, 64  # 32x64 = 2048
            elif feature_dim == 65536:
                h, w = 256, 256  # 256x256 = 65536
            else:
                side_dim = int(feature_dim ** 0.5)
                if side_dim * side_dim == feature_dim:
                    h, w = side_dim, side_dim
                else:
                    h = side_dim
                    w = feature_dim // h
                    if h * w != feature_dim:
                        h, w = side_dim, side_dim
                        img_features = img_features[:, :h*w]
                        
            img_per_camera = img_features.view(batch_size, 1, h, w)
            if img_per_camera.shape[1] == 1:
                img_per_camera = img_per_camera.expand(-1, 3, -1, -1)

            batch = {
                'observation.images.camera1': img_per_camera,
                'observation.images.camera2': img_per_camera,
                'observation.images.camera3': img_per_camera,
                'observation.state': state if state is not None else torch.zeros(batch_size, 8).to(img_features.device),
                'observation.language.tokens': torch.zeros(batch_size, 1, dtype=torch.long).to(img_features.device),
                'observation.language.attention_mask': torch.ones(batch_size, 1, dtype=torch.bool).to(img_features.device)
            }
            
            try:
                actions = self.model.select_action(batch)
                if actions.shape[0] == 1 and batch_size > 1:
                    # If teacher returns only one action for a batch, expand it 
                    # (This might happen if the policy is optimized for single-step inference)
                    actions = actions.expand(batch_size, -1)
                return actions
            except Exception as e:
                logger.warning(f"Real model failed: {e}, falling back to simulated")
                if hasattr(self, 'encoder') and hasattr(self, 'transformer') and hasattr(self, 'action_head'):
                    x = self.encoder(img_features)
                    x = x.unsqueeze(1)
                    x = self.transformer(x)
                    x = x.squeeze(1)
                    return self.action_head(x)
                else:
                    return torch.zeros(img_features.shape[0], 14).to(img_features.device)
        else:
            x = self.encoder(img_features)
            x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)
            return self.action_head(x)
    
    def get_features(self, x):
        if self.use_real and self.model is not None:
            with torch.no_grad():
                encoded = self.model.backbone(x) if hasattr(self.model, 'backbone') else x
                return encoded
        else:
            x = self.encoder(x)
            x = x.unsqueeze(1)
            x = self.transformer(x)
            return x.squeeze(1)
    
    def get_num_parameters(self):
        if self.use_real and self.model is not None:
            return sum(p.numel() for p in self.model.parameters())
        else:
            return sum(p.numel() for p in self.parameters())
    
    def enable_gradient_checkpointing(self):
        if self.use_real and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for teacher model")
        
    def disable_gradient_checkpointing(self):
        if self.use_real and hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled for teacher model")
    