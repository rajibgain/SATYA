import os
import glob
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class AIGeneratedDataset(Dataset):
    """
    Dataset loader for AI-generated vs Real image classification.
    Expects structure:
        root/
            real/
            ai_generated/
    Class mapping:
        0 = real
        1 = ai_generated
    """
    def __init__(self, root_dir, transform=None, skip_corrupt=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.skip_corrupt = skip_corrupt
        self.samples = []
        
        self.class_map = {
            'real': 0,
            'ai_generated': 1
        }
        
        # We also support metadata in the future
        # (image_path, label, source, generator, group_id)
        # For now we just load paths from the directory structure
        if self.root_dir.exists():
            self._scan_directory()
            
    def _scan_directory(self):
        for class_name, label in self.class_map.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            # Find common image formats
            all_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                all_paths.extend(class_dir.rglob(ext))
                
            for img_path in sorted(all_paths):
                # (path, label, source, generator, group_id)
                # Currently setting unknown for future metadata
                self.samples.append({
                    'path': str(img_path),
                    'label': label,
                    'source': 'unknown',
                    'generator': 'unknown',
                    'group_id': 'unknown'
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        label = sample['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError) as e:
            if self.skip_corrupt:
                # Fallback to another index if corrupt
                new_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(new_idx)
            else:
                raise RuntimeError(f"Failed to load {img_path}") from e
                
        if self.transform:
            image = self.transform(image)
            
        return image, label, sample
