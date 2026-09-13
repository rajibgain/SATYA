"""
FakeShield - AI Dataset Access Test

This script performs a MINIMAL access test for a Hugging Face dataset.
It uses 'streaming' mode to inspect the dataset structure and extracts exactly two
samples (one of each class) without downloading the entire dataset.

This ensures we can access and decode the images locally before committing to a 
full download or writing a heavy processing script.
"""

import os
from pathlib import Path
import sys

try:
    # Hugging Face datasets library is used to access the remote dataset metadata and stream it
    from datasets import load_dataset_builder, load_dataset
except ImportError:
    print("[ERROR] The 'datasets' package is not installed.")
    print("Please install it using: pip install datasets")
    sys.exit(1)

def main():
    # The target Hugging Face dataset repository
    dataset_name = "Shanmuk4622/ai-image-detection-dataset"
    print("=" * 60)
    print(f"Testing access to Hugging Face dataset: {dataset_name}")
    print("=" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 1. Inspect dataset configuration and features without downloading anything
    # -------------------------------------------------------------------------
    try:
        # load_dataset_builder fetches the dataset metadata (schema, splits, etc.)
        builder = load_dataset_builder(dataset_name)
    except Exception as e:
        # If the requested dataset/configuration cannot be accessed using the current 
        # API, we must not guess the configuration name. We catch the exact error and print it.
        # The library usually prints the available options automatically in the exception message.
        print(f"[ERROR] Failed to access dataset: {dataset_name}")
        print(f"Exact error:\n{e}\n")
        print("Note: Do not guess the configuration name. Use the exact configuration if required.")
        return

    print("--- Dataset Structure Information ---")
    
    # Configurations (subsets)
    config_name = builder.info.config_name
    print(f"Configuration (Subset): {config_name if config_name else 'Default / None'}")
    
    # Features/Columns
    features = builder.info.features
    print("\nColumns/Features:")
    if features:
        for fname, ftype in features.items():
            print(f"  - {fname}: {ftype}")
    else:
        print("  - N/A")
        
    # Splits and Record Counts
    splits = builder.info.splits
    split_to_use = "train"
    print("\nAvailable Splits:")
    if splits:
        for sname, sinfo in splits.items():
            print(f"  - {sname}: {sinfo.num_examples} records")
        # Ensure we use a valid split for streaming just in case 'train' doesn't exist
        if "train" not in splits:
            split_to_use = list(splits.keys())[0]
    else:
        print("  - N/A")
        
    print("-" * 60 + "\n")
    
    # -------------------------------------------------------------------------
    # 2. Stream dataset to get exactly one real and one AI-generated sample
    # -------------------------------------------------------------------------
    print(f"Streaming dataset (split='{split_to_use}') to extract minimal samples...")
    try:
        # streaming=True means we download images on-the-fly, one by one.
        dataset_stream = load_dataset(dataset_name, split=split_to_use, streaming=True)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset stream: {e}")
        return

    # We want exactly two different classes (e.g., Real and AI)
    samples_by_label = {}
    
    try:
        for record in dataset_stream:
            # Most image classification datasets have a "label" column
            if "label" not in record:
                print(f"[ERROR] Could not find a 'label' column in the record. Found: {list(record.keys())}")
                return
                
            lbl = record["label"]
            # Store the first record we find for each label
            if lbl not in samples_by_label:
                samples_by_label[lbl] = record
                
            # Stop streaming as soon as we have one sample from each class
            if len(samples_by_label) >= 2:
                break
    except Exception as e:
        print(f"[ERROR] Failed while iterating the dataset stream: {e}")
        return

    if len(samples_by_label) < 2:
        print("[ERROR] Could not find two distinct classes (Real/AI) in the stream.")
        return

    # Output directory for temporarily saving images to prove they decode successfully
    out_dir = Path("outputs/ai_dataset_access_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary output directory created at: {out_dir}\n")

    # -------------------------------------------------------------------------
    # 3. Print details and verify decoding with PIL
    # -------------------------------------------------------------------------
    def process_sample(sample_record, display_name):
        print(f"--- {display_name} Sample ---")
        
        # Label handling
        label_id = sample_record["label"]
        label_str = str(label_id)
        # Try to resolve class name if ClassLabel mapping is present in dataset features
        if features and "label" in features and hasattr(features["label"], "names"):
            try:
                label_str = f"{label_id} ({features['label'].names[label_id]})"
            except IndexError:
                pass
        print(f"Label/Class: {label_str}")
        
        # Generator/Source handling
        if "generator" in sample_record:
            print(f"Generator/Source: {sample_record['generator']}")
        else:
            print("Generator/Source: N/A (Not provided in dataset)")
            
        # The 'image' feature is typically a PIL.Image object when decoded by the datasets library
        img = sample_record.get("image")
        if img:
            print(f"Image Type: {type(img)}")
            print(f"Image Size: {img.size}")
            print(f"Image Mode: {img.mode}")
            
            # Save the image to verify PIL decoding works properly
            save_path = out_dir / f"{display_name.lower().replace(' ', '_')}_sample.jpg"
            try:
                # Convert to RGB in case it's grayscale or RGBA to ensure it saves as JPG correctly
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(save_path)
                print(f"-> Successfully decoded and temporarily saved to: {save_path}")
                return True
            except Exception as e:
                print(f"-> [ERROR] Failed to save/decode image: {e}")
                return False
        else:
            print("-> [ERROR] No 'image' data found in record.")
            return False

    # Identify the labels. In many datasets 0=Real, 1=Fake (or vice-versa). 
    # If the feature names exist, we use them to accurately distinguish real vs AI.
    lbl_keys = sorted(list(samples_by_label.keys()))
    
    real_sample_ok = False
    ai_sample_ok = False
    
    for lbl_id in lbl_keys:
        sample = samples_by_label[lbl_id]
        
        # Try to guess which class is Real and which is AI by the feature name
        name_hint = ""
        if features and "label" in features and hasattr(features["label"], "names"):
            try:
                name_hint = features["label"].names[lbl_id].lower()
            except IndexError:
                pass
                
        if "real" in name_hint or "nature" in name_hint or "authentic" in name_hint:
            sample_type = "Real"
        elif "ai" in name_hint or "fake" in name_hint or "generated" in name_hint:
            sample_type = "AI"
        else:
            # Fallback if no names are provided, we just map 0 to Real and 1 to AI generically
            sample_type = "Real" if lbl_id == 0 else "AI"
            
        print("")
        ok = process_sample(sample, sample_type)
        if sample_type == "Real":
            real_sample_ok = ok
        elif sample_type == "AI":
            ai_sample_ok = ok
            
    # -------------------------------------------------------------------------
    # 4. Final Verification Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30)
    print("SUMMARY")
    print("=" * 30)
    print("[OK] Dataset access")
    
    if real_sample_ok:
        print("[OK] Real sample decoded")
    else:
        print("[FAIL] Real sample decoded")
        
    if ai_sample_ok:
        print("[OK] AI sample decoded")
    else:
        print("[FAIL] AI sample decoded")
        
    print("=" * 30)

if __name__ == "__main__":
    main()
