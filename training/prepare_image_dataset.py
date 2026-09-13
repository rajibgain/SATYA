"""
FakeShield - Dataset Preparation Module
Downloads and prepares the CIFAKE image dataset from Hugging Face for training.

NOTE: 
CIFAKE is for AI-generated-vs-real image detection and should not be presented 
as a universal deepfake detector (e.g., face swapping, video manipulation).

This script extracts a small, balanced, reproducible subset from the official splits
to quickly prototype the EfficientNet-B0 model.
"""

import argparse
import random
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: The 'datasets' package is required but not installed.")
    print("Please install it using: pip install datasets")
    exit(1)

# Define the exact number of images needed for each class in each split
TARGET_COUNTS = {
    'train': {'real': 2000, 'fake': 2000},
    'val': {'real': 500, 'fake': 500},
    'test': {'real': 500, 'fake': 500}
}

# The target output directory where images will be saved
BASE_OUT_DIR = Path("data/processed/image")

def check_existing_images(force: bool):
    """
    Check if any of the target folders already contain images.
    If they do, require the --force flag to proceed and overwrite.
    """
    found_images = False
    
    # Loop through all our target directories to check for existing JPG/PNG files
    for split in TARGET_COUNTS.keys():
        for label in ['real', 'fake']:
            dir_path = BASE_OUT_DIR / split / label
            if dir_path.exists():
                images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
                if len(images) > 0:
                    found_images = True
                    break
        if found_images:
            break
            
    if found_images and not force:
        print("Error: Target folders already contain images.")
        print("Use the --force command-line option to overwrite them.")
        exit(1)
    
    if found_images and force:
        print("Warning: --force is active. Existing images will be overwritten/ignored.")

def get_label_indices(dataset_split):
    """
    Determine the numerical indices for 'real' and 'fake' classes from the Hugging Face dataset.
    Defaults to 0=fake, 1=real based on typical CIFAKE structures if metadata is missing.
    """
    label_feature = dataset_split.features['label']
    label_names = label_feature.names if hasattr(label_feature, 'names') else ['FAKE', 'REAL']
    label_names_lower = [name.lower() for name in label_names]
    
    fake_idx = label_names_lower.index('fake')
    real_idx = label_names_lower.index('real')
    
    return real_idx, fake_idx

def save_images(images, label_name, split_dir_name):
    """
    Helper function to save a list of PIL Images to the specified folder with sequential names.
    """
    out_dir = BASE_OUT_DIR / split_dir_name / label_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Saving {len(images)} {label_name} images to {out_dir}...")
    for i, img in enumerate(images):
        # Convert to RGB to ensure compatibility with JPEG format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save sequentially as JPG
        filename = f"{label_name}_{i:04d}.jpg"
        img.save(out_dir / filename, format="JPEG")

def process_test_split(official_test, seed: int):
    """
    Process the official test set to create our FakeShield test subset.
    """
    print("\nProcessing official test split for FakeShield test split...")
    real_idx, fake_idx = get_label_indices(official_test)
    
    real_items = []
    fake_items = []
    
    print("  Filtering test dataset by label...")
    for item in official_test:
        if item['label'] == real_idx:
            real_items.append(item['image'])
        elif item['label'] == fake_idx:
            fake_items.append(item['image'])
            
    # Deterministic shuffle to ensure reproducibility
    rng = random.Random(seed)
    rng.shuffle(real_items)
    rng.shuffle(fake_items)
    
    # Slice the required target counts
    test_real = real_items[:TARGET_COUNTS['test']['real']]
    test_fake = fake_items[:TARGET_COUNTS['test']['fake']]
    
    save_images(test_real, "real", "test")
    save_images(test_fake, "fake", "test")

def main():
    parser = argparse.ArgumentParser(description="Prepare a small reproducible CIFAKE subset for FakeShield.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing images if folders are not empty.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection.")
    args = parser.parse_args()
    
    print("=" * 50)
    print("FakeShield Image Dataset Preparation")
    print("=" * 50)
    print("Target dataset: dragonintelligence/CIFAKE-image-dataset")
    print("This script will download the dataset and extract:")
    print(f"  Train: {TARGET_COUNTS['train']['real']} real / {TARGET_COUNTS['train']['fake']} fake")
    print(f"  Val:   {TARGET_COUNTS['val']['real']} real / {TARGET_COUNTS['val']['fake']} fake")
    print(f"  Test:  {TARGET_COUNTS['test']['real']} real / {TARGET_COUNTS['test']['fake']} fake")
    print(f"Using random seed: {args.seed}")
    print("=" * 50)
    
    # 1. Check if files already exist to prevent accidental overwrites
    check_existing_images(args.force)
    
    # 2. Download the dataset
    print("\nConnecting to Hugging Face and downloading dataset (this may take a while)...")
    try:
        dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        exit(1)
        
    print("\nDataset downloaded successfully!")
    print(dataset)
    
    if 'train' not in dataset or 'test' not in dataset:
        print("Error: Expected 'train' and 'test' splits in the official dataset.")
        exit(1)
        
    official_train = dataset['train']
    official_test = dataset['test']
    
    # 3. Process the official train split to create our 'train' and 'val' subsets
    # We do this to avoid mixing test data into the training process.
    print("\nProcessing official train split to create FakeShield train and val splits...")
    real_idx, fake_idx = get_label_indices(official_train)
    
    real_items = []
    fake_items = []
    
    print("  Filtering train dataset by label...")
    for item in official_train:
        if item['label'] == real_idx:
            real_items.append(item['image'])
        elif item['label'] == fake_idx:
            fake_items.append(item['image'])
            
    # Deterministic shuffle
    rng = random.Random(args.seed)
    rng.shuffle(real_items)
    rng.shuffle(fake_items)
    
    # Slice out the training images
    train_real = real_items[:TARGET_COUNTS['train']['real']]
    train_fake = fake_items[:TARGET_COUNTS['train']['fake']]
    
    # Slice out the validation images right after the training ones
    val_real_start = TARGET_COUNTS['train']['real']
    val_fake_start = TARGET_COUNTS['train']['fake']
    
    val_real = real_items[val_real_start : val_real_start + TARGET_COUNTS['val']['real']]
    val_fake = fake_items[val_fake_start : val_fake_start + TARGET_COUNTS['val']['fake']]
    
    # Save the splits
    save_images(train_real, "real", "train")
    save_images(train_fake, "fake", "train")
    
    save_images(val_real, "real", "val")
    save_images(val_fake, "fake", "val")
    
    # 4. Process the test split
    process_test_split(official_test, args.seed)
    
    # 5. Print final verification counts
    print("\n" + "=" * 50)
    print("Dataset Preparation Complete!")
    print("=" * 50)
    print("Final image counts per folder:")
    
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            dir_path = BASE_OUT_DIR / split / label
            if dir_path.exists():
                count = len(list(dir_path.glob("*.jpg")))
                print(f"  {dir_path}: {count} images")
            else:
                print(f"  {dir_path}: 0 images")

if __name__ == "__main__":
    main()
