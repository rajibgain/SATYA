import os
import sys
import shutil
import zipfile
import pandas as pd
from pathlib import Path

# Use absolute path to ensure robustness from any working directory
ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "data/raw/ntire2026/metadata/labels.csv"
ZIP_PATH = ROOT_DIR / "data/raw/ntire2026/shard_5.zip"
OUT_DIR = ROOT_DIR / "data/processed/image_ntire"
MANIFEST_PATH = OUT_DIR / "manifest.csv"

def check_existing_dataset():
    """Verify if the dataset already exists and is valid."""
    if not MANIFEST_PATH.exists():
        return False
        
    print("Found existing manifest. Verifying dataset...")
    try:
        df = pd.read_csv(MANIFEST_PATH)
    except Exception:
        return False
        
    if len(df) != 10000:
        return False
        
    # verify splits
    split_counts = df['split'].value_counts()
    if split_counts.get('train', 0) != 8000: return False
    if split_counts.get('val', 0) != 1000: return False
    if split_counts.get('test', 0) != 1000: return False
    
    # check if files exist
    for _, row in df.iterrows():
        p = ROOT_DIR / row['path']
        if not p.exists():
            return False
            
    return True

def print_report(source_rows, manifest_df):
    missing_files = 0
    invalid_extensions = 0
    invalid_labels = 0

    train_real = train_fake = 0
    val_real = val_fake = 0
    test_real = test_fake = 0

    for _, row in manifest_df.iterrows():
        p = ROOT_DIR / row['path']
        if not p.exists():
            missing_files += 1
        if not row['image_name'].lower().endswith('.jpg'):
            invalid_extensions += 1
        if row['label'] not in [0, 1]:
            invalid_labels += 1
            
        s = row['split']
        l = row['label']
        if s == 'train':
            if l == 0: train_real += 1
            else: train_fake += 1
        elif s == 'val':
            if l == 0: val_real += 1
            else: val_fake += 1
        elif s == 'test':
            if l == 0: test_real += 1
            else: test_fake += 1

    dup_names = len(manifest_df) - len(manifest_df['image_name'].unique())

    print("\nNTIRE DATASET PREPARATION")
    print("=========================")
    print(f"Source rows: {source_rows}")
    print(f"Selected real: {train_real + val_real + test_real}")
    print(f"Selected AI: {train_fake + val_fake + test_fake}")
    print(f"Total selected: {len(manifest_df)}\n")

    print("Train:")
    print(f"  Real: {train_real}")
    print(f"  Fake: {train_fake}")
    print(f"  Total: {train_real + train_fake}\n")

    print("Validation:")
    print(f"  Real: {val_real}")
    print(f"  Fake: {val_fake}")
    print(f"  Total: {val_real + val_fake}\n")

    print("Test:")
    print(f"  Real: {test_real}")
    print(f"  Fake: {test_fake}")
    print(f"  Total: {test_real + test_fake}\n")

    print("Verification:")
    print(f"  Missing files: {missing_files}")
    print(f"  Duplicate filenames: {dup_names}")
    print(f"  Invalid labels: {invalid_labels}")
    print(f"  Invalid extensions: {invalid_extensions}")
    
    if missing_files == 0 and dup_names == 0 and invalid_labels == 0 and invalid_extensions == 0:
        if (train_real == 4000 and train_fake == 4000 and 
            val_real == 500 and val_fake == 500 and 
            test_real == 500 and test_fake == 500):
            print("\nSUCCESS: Dataset is perfectly validated and ready for use.")
        else:
            print("\nWARNING: Counts do not match expected balances.")
    else:
        print("\nERROR: Validation failed.")
        sys.exit(1)

def main():
    if check_existing_dataset():
        print("Valid processed dataset found! Skipping extraction.")
        df = pd.read_csv(MANIFEST_PATH)
        # In this branch, source rows is technically unknown without re-reading CSV, 
        # but 27643 is the hardcoded known value from the source dataset info.
        print_report(27643, df)
        return
        
    print("Starting NTIRE dataset preparation...")
    
    if not CSV_PATH.exists():
        print(f"Error: labels CSV is missing at {CSV_PATH}")
        sys.exit(1)
    if not ZIP_PATH.exists():
        print(f"Error: ZIP archive is missing at {ZIP_PATH}")
        sys.exit(1)
        
    print(f"Loading metadata from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    source_rows = len(df)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    if df['image_name'].isna().any() or df['label'].isna().any():
        print("Error: Missing values in image_name or label column.")
        sys.exit(1)
        
    if not set(df['label'].unique()).issubset({0, 1}):
        print("Error: Labels must be exactly 0 (Real) and 1 (AI).")
        sys.exit(1)
        
    print("Sampling a balanced subset (Seed: 42)...")
    real_df = df[df['label'] == 0].sample(n=5000, random_state=42)
    fake_df = df[df['label'] == 1].sample(n=5000, random_state=42)
    
    # Train/Val/Test splits
    train_real = real_df.iloc[:4000].copy()
    val_real = real_df.iloc[4000:4500].copy()
    test_real = real_df.iloc[4500:5000].copy()

    train_fake = fake_df.iloc[:4000].copy()
    val_fake = fake_df.iloc[4000:4500].copy()
    test_fake = fake_df.iloc[4500:5000].copy()
    
    splits = [
        (train_real, 'train'), (val_real, 'val'), (test_real, 'test'),
        (train_fake, 'train'), (val_fake, 'val'), (test_fake, 'test')
    ]
    
    for split_df, split_name in splits:
        split_df['split'] = split_name
        
    final_df = pd.concat([s[0] for s in splits]).reset_index(drop=True)
    
    print("Creating directory structure...")
    for split in ['train', 'val', 'test']:
        for label_str in ['real', 'fake']:
            (OUT_DIR / split / label_str).mkdir(parents=True, exist_ok=True)
            
    print(f"Extracting 10,000 images from {ZIP_PATH}...")
    manifest_records = []
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zip_files_set = set(zf.namelist())
            
            for idx, row in final_df.iterrows():
                if idx % 1000 == 0 and idx > 0:
                    print(f"  Extracted {idx}/10000 images...")
                    
                img_name = row['image_name']
                lbl = row['label']
                split = row['split']
                lbl_str = 'real' if lbl == 0 else 'fake'
                
                zip_member = f"shard_5/images/{img_name}"
                if zip_member not in zip_files_set:
                    print(f"\nError: {zip_member} not found in zip archive!")
                    sys.exit(1)
                    
                target_path = OUT_DIR / split / lbl_str / img_name
                rel_path = f"data/processed/image_ntire/{split}/{lbl_str}/{img_name}"
                
                # Perform the extraction safely to the flat target_path
                with zf.open(zip_member) as zf_file:
                    with open(target_path, 'wb') as out_file:
                        shutil.copyfileobj(zf_file, out_file)
                        
                manifest_records.append({
                    'image_name': img_name,
                    'label': lbl,
                    'split': split,
                    'path': rel_path
                })
    except Exception as e:
        print(f"Error during zip extraction: {e}")
        sys.exit(1)
        
    print("Extraction complete. Writing manifest...")
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    
    print("Verifying outputs...")
    print_report(source_rows, manifest_df)

if __name__ == "__main__":
    main()
