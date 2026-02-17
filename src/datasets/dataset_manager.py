import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import json
from typing import Dict, List

class DatasetManager:
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.available_datasets = [
            "lerobot/libero",
            "lerobot/aloha_mobile_cabinet", 
            "lerobot/aloha_static_candy",
            "lerobot/pusht"
        ]
        self.downloaded_datasets = {}
        
    def list_available_datasets(self) -> List[str]:
        print("Available LeRobot datasets:")
        for i, dataset in enumerate(self.available_datasets, 1):
            print(f"  {i}. {dataset}")
        return self.available_datasets
    
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> LeRobotDataset:
        print(f"\nDownloading dataset: {dataset_name}")
        
        dataset_path = os.path.join(self.data_dir, dataset_name.replace("/", "_"))
        
        if os.path.exists(dataset_path) and not force_redownload:
            print(f"Dataset already exists at {dataset_path}")
            print("Loading existing dataset...")
        else:
            print(f"Downloading to {dataset_path}...")
            os.makedirs(dataset_path, exist_ok=True)
        
        try:
            dataset = LeRobotDataset(dataset_name)
            self.downloaded_datasets[dataset_name] = {
                "dataset": dataset,
                "path": dataset_path,
                "info": self._get_dataset_info(dataset)
            }
            
            print(f"✅ Successfully loaded dataset: {dataset_name}")
            self._print_dataset_info(dataset_name)
            
            return dataset
            
        except Exception as e:
            print(f"❌ Failed to download/load dataset {dataset_name}: {e}")
            return None
    
    def _get_dataset_info(self, dataset: LeRobotDataset) -> Dict:
        try:
            return {
                "num_episodes": len(dataset.meta.episodes) if hasattr(dataset.meta, 'episodes') else "Unknown",
                "total_frames": len(dataset) if hasattr(dataset, '__len__') else "Unknown",
                "features": list(dataset[0].keys()) if len(dataset) > 0 else [],
                "camera_views": self._detect_camera_views(dataset)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_camera_views(self, dataset: LeRobotDataset) -> List[str]:
        try:
            if len(dataset) > 0:
                sample = dataset[0]
                camera_views = [key for key in sample.keys() if 'image' in key.lower() or 'cam' in key.lower()]
                return camera_views
            return []
        except:
            return []
    
    def _print_dataset_info(self, dataset_name: str):
        if dataset_name in self.downloaded_datasets:
            info = self.downloaded_datasets[dataset_name]["info"]
            print(f"\nDataset Information for {dataset_name}:")
            print(f"  Episodes: {info.get('num_episodes', 'Unknown')}")
            print(f"  Total frames: {info.get('total_frames', 'Unknown')}")
            print(f"  Features: {', '.join(info.get('features', []))}")
            print(f"  Camera views: {', '.join(info.get('camera_views', []))}")
    
    def download_sample_datasets(self) -> Dict[str, LeRobotDataset]:
        print("="*60)
        print("DOWNLOADING SAMPLE DATASETS")
        print("="*60)
        
        results = {}
        
        test_datasets = ["lerobot/pusht"]
        
        for dataset_name in test_datasets:
            dataset = self.download_dataset(dataset_name)
            if dataset is not None:
                results[dataset_name] = dataset
                
        return results
    
    def create_dataset_loader(self, dataset_name: str) -> callable:
        if dataset_name not in self.downloaded_datasets:
            raise ValueError(f"Dataset {dataset_name} not downloaded")
            
        dataset = self.downloaded_datasets[dataset_name]["dataset"]
        
        def load_episode(episode_index: int):
            try:
                from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
                to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
                
                episode_data = []
                for idx in range(from_idx, to_idx):
                    frame = dict(dataset[idx])
                    episode_data.append(frame)
                
                return {
                    "episode_index": episode_index,
                    "from_index": from_idx,
                    "to_index": to_idx,
                    "length": len(episode_data),
                    "data": episode_data
                }
            except Exception as e:
                print(f"Error loading episode {episode_index}: {e}")
                return None
        
        return load_episode
    
    def save_dataset_manifest(self, filepath: str = "results/dataset_manifest.json"):
        manifest = {
            "data_directory": self.data_dir,
            "downloaded_datasets": {
                name: {
                    "path": info["path"],
                    "info": info["info"]
                }
                for name, info in self.downloaded_datasets.items()
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        print(f"Dataset manifest saved to: {filepath}")

if __name__ == "__main__":
    manager = DatasetManager()
    manager.list_available_datasets()
    datasets = manager.download_sample_datasets()
    manager.save_dataset_manifest()
