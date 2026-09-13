import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import io

class JPEGCompression:
    """
    Custom transform that applies JPEG compression with a random quality
    to prevent the network from memorizing lossless vs lossy artifacts.
    """
    def __init__(self, quality_lower=50, quality_upper=100):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        quality = random.randint(self.quality_lower, self.quality_upper)
        
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        
        return Image.open(buffer).convert("RGB")

def get_transforms(train=True):
    """
    Returns the torchvision transforms for AI-generated image detection.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            JPEGCompression(50, 95),  # Apply random JPEG compression
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

class AIDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")
        
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(real_dir, filename))
                    self.labels.append(0)  # 0 for Real
                    
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(fake_dir, filename))
                    self.labels.append(1)  # 1 for AI-Generated

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback to random noise image if corrupt (rare but happens in large datasets)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
