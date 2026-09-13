import sys
from PIL import Image

try:
    image = Image.open("data/processed/image_ntire/test/fake/0081a9d39a3e5618cd72.jpg").convert('RGB')
    print("Image loaded")
    
    from src.ai_detection.inference import analyze_ai_generation
    print("Inference imported")
    
    res = analyze_ai_generation(image)
    print("Result:", res)
    
except Exception as e:
    import traceback
    traceback.print_exc()
