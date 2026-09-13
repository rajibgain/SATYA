import sys
import platform
import subprocess
from pathlib import Path

def run_diagnostics():
    print("=" * 40)
    print("SATYA - Environment Diagnostics")
    print("=" * 40)
    
    all_ready = True
    
    # 1. Python version
    try:
        python_version = sys.version.split()[0]
        print(f"Python Version: {python_version}")
    except Exception as e:
        print(f"Python Version: Error ({e})")
        
    # 2. Python executable path
    try:
        print(f"Python Executable: {sys.executable}")
    except Exception as e:
        print(f"Python Executable: Error ({e})")
    
    # 3. Virtual environment active
    try:
        is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        print(f"Virtual Environment Active: {is_venv}")
        if not is_venv:
            print("[FAIL] Virtual Environment")
            all_ready = False
        else:
            print("[OK] Virtual Environment")
    except Exception as e:
        print(f"[FAIL] Virtual Environment (Error: {e})")
        all_ready = False
    
    print("-" * 40)
    
    def check_module(module_name, print_name, check_version=True):
        try:
            mod = __import__(module_name)
            if check_version:
                version = getattr(mod, '__version__', 'unknown')
                print(f"[OK] {print_name} (version: {version})")
            else:
                print(f"[OK] {print_name}")
            return True
        except Exception:
            print(f"[FAIL] {print_name}")
            return False

    # 4. PyTorch version and import
    if not check_module('torch', 'PyTorch'):
        all_ready = False
        
    # 5. Torchvision version and import
    if not check_module('torchvision', 'Torchvision'):
        all_ready = False
        
    # 6. CUDA availability
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"[INFO] CUDA available: {cuda_avail}")
    except Exception:
        print("[INFO] CUDA available: False")
        
    # 7 to 15. Other dependencies
    deps = [
        ('transformers', 'Transformers', True),
        ('datasets', 'Datasets', True),
        ('accelerate', 'Accelerate', True),
        ('cv2', 'OpenCV', True),
        ('skimage', 'scikit-image', True),
        ('sklearn', 'scikit-learn', True),
        ('librosa', 'Librosa', True),
        ('soundfile', 'SoundFile', True),
        ('pytorch_grad_cam', 'Grad-CAM', False),
    ]
    
    for mod_name, print_name, chk_ver in deps:
        if not check_module(mod_name, print_name, chk_ver):
            all_ready = False
            
    # 16. FFmpeg availability
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("[OK] FFmpeg")
    except Exception:
        print("[FAIL] FFmpeg")
        all_ready = False

    print("-" * 40)
    
    # Project directories
    directories = [
        "data/raw",
        "data/processed",
        "data/demo_samples",
        "models/text",
        "models/image",
        "models/audio",
        "outputs",
        "logs",
        "tests"
    ]
    
    base_dir = Path.cwd()
    for d in directories:
        try:
            dir_path = base_dir / d
            if dir_path.is_dir():
                print(f"[OK] {d}")
            else:
                print(f"[FAIL] {d}")
                all_ready = False
        except Exception:
            print(f"[FAIL] {d}")
            all_ready = False
            
    print("\n========================================")
    print("SATYA Environment Status")
    print("========================================")
    print()
    if all_ready:
        print("[READY] Core environment is ready.")
    else:
        print("[NOT READY] Fix the failed checks above.")

if __name__ == "__main__":
    run_diagnostics()
