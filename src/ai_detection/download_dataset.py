import os
import argparse
from datasets import load_dataset
from PIL import Image

def download_dataset(total_images=2000, dataset_name="TheKernel01/Tiny-GenImage", split="train"):
    """
    Downloads a subset of the dataset.
    total_images will be split 50/50 between real and fake.
    """
    output_dir = os.path.join("data", "processed", "ai_detection")
    real_dir = os.path.join(output_dir, "train", "real")
    fake_dir = os.path.join(output_dir, "train", "fake")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print(f"Loading {dataset_name} in streaming mode...")
    try:
        # Load dataset in streaming mode to avoid downloading everything
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    target_per_class = total_images // 2
    real_count = 0
    fake_count = 0

    print(f"Targeting {target_per_class} real and {target_per_class} fake images.")

    for i, item in enumerate(dataset):
        if real_count >= target_per_class and fake_count >= target_per_class:
            break

        try:
            # TheKernel01/Tiny-GenImage has 'image' and 'label'
            img = item['image']
            label = item['label']

            # If img is not a PIL Image, convert it
            if not isinstance(img, Image.Image):
                continue
                
            # Convert to RGB to ensure consistency
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if label == 0 and real_count < target_per_class:
                save_path = os.path.join(real_dir, f"real_{real_count}.jpg")
                img.save(save_path, "JPEG", quality=95)
                real_count += 1
                if real_count % 100 == 0:
                    print(f"Downloaded {real_count}/{target_per_class} real images.")
            
            elif label == 1 and fake_count < target_per_class:
                save_path = os.path.join(fake_dir, f"fake_{fake_count}.jpg")
                img.save(save_path, "JPEG", quality=95)
                fake_count += 1
                if fake_count % 100 == 0:
                    print(f"Downloaded {fake_count}/{target_per_class} fake images.")

        except Exception as e:
            # Skip corrupted images
            pass

    print(f"\nDownload complete. Real: {real_count}, Fake: {fake_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AI detection dataset subset.")
    parser.add_argument("--total", type=int, default=1000, help="Total number of images to download")
    args = parser.parse_args()
    
    download_dataset(args.total)
