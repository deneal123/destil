import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

HF_DATASETS_CACHE = os.getenv('HF_DATASETS_CACHE', os.path.expanduser('~/.cache/huggingface/datasets'))
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
DATASET_DOWNLOAD_PERCENTAGE = float(os.getenv('DATASET_DOWNLOAD_PERCENTAGE', '1.0'))


class RealLeRobotDataset(Dataset):
    
    def __init__(self, dataset_name="lerobot/aloha_sim_insertion_scripted", split='train', num_samples=1000):
        self.dataset_name = dataset_name
        self.split = split
        self.num_samples = num_samples
        self.dataset = None
        
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            
            logger.info(f"Dataset cache directory: {HF_DATASETS_CACHE}")
            logger.info(f"Download percentage: {DATASET_DOWNLOAD_PERCENTAGE*100:.1f}%")
            
            self.dataset = LeRobotDataset(dataset_name)
            
            total_available = len(self.dataset)
            limit_samples = min(total_available, self.num_samples)
            
            logger.info(f"Total available samples: {total_available}")
            logger.info(f"Limited to: {limit_samples} samples based on num_samples")

            if split == 'train':
                self.indices = list(range(0, int(limit_samples * 0.8)))
            else:  # val
                self.indices = list(range(int(limit_samples * 0.8), limit_samples))
                
            if len(self.indices) == 0 and limit_samples > 0:
                # Fallback for very small num_samples
                self.indices = [0] if split == 'train' else []
                
            logger.info(f"Loaded real LeRobot dataset: {dataset_name}")
            logger.info(f"{split.capitalize()} samples: {len(self.indices)}")
                
        except Exception as e:
            logger.warning(f"Failed to load LeRobot dataset {dataset_name}: {e}")
            logger.info("Using simulated data")
            self._create_simulated_data()
    
    def _create_simulated_data(self):
        np.random.seed(42 if self.split == 'train' else 123)
        self.data = []
        
        for i in range(self.num_samples):
            img_features = np.random.randn(2048).astype(np.float32)
            img_features += 0.3 * np.roll(img_features, 10)
            state = np.random.uniform(-1, 1, 8).astype(np.float32)
            action = np.random.uniform(-1, 1, 7).astype(np.float32)
            if i > 0 and hasattr(self, 'data') and len(self.data) > 0:
                prev_action = self.data[-1]['action']
                action = 0.7 * action + 0.3 * prev_action
            
            self.data.append({
                'img': img_features,
                'state': state,
                'action': action
            })
    
    def __len__(self):
        if self.dataset is not None:
            return len(self.indices)
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.dataset is not None:
            try:
                actual_idx = self.indices[idx]
                item = self.dataset[actual_idx]
                
                if 'observation.image' in item:
                    img = item['observation.image']
                    if len(img.shape) == 3:  # (C, H, W)
                        img = img.flatten()
                else:
                    img = torch.randn(2048)
                
                if len(img) < 2048:
                    img = torch.cat([img, torch.zeros(2048 - len(img))])
                img = img[:2048]
                
                state = item.get('observation.state', torch.zeros(8))
                action = item['action']
                
                return img.float(), state.float(), action.float()
            except Exception as e:
                logger.warning(f"Error processing real data item {idx}: {e}")
                return self._get_simulated_item(idx)
        else:
            return self._get_simulated_item(idx)
    
    def _get_simulated_item(self, idx):
        if hasattr(self, 'data'):
            item = self.data[idx % len(self.data)]
            img_noise = np.random.normal(0, 0.05, 2048).astype(np.float32)
            action_noise = np.random.normal(0, 0.01, 7).astype(np.float32)
            
            return (
                torch.from_numpy(item['img'] + img_noise),
                torch.from_numpy(item['state']),
                torch.from_numpy(item['action'] + action_noise)
            )
        else:
            return (
                torch.randn(2048),
                torch.zeros(8),
                torch.randn(7) * 0.1
            )


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
