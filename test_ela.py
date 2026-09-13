import requests
from PIL import Image
import os

# 1. Known real NTIRE image
real_image_path = 'data/processed/image_ntire/test/real/0016ed96770dc80ff1b5.jpg'
# 2. Known fake NTIRE image
fake_image_path = 'data/processed/image_ntire/test/fake/0081a9d39a3e5618cd72.jpg'
# 3. Normal PNG
png_image_path = 'test_normal.png'
if not os.path.exists(png_image_path):
    Image.new('RGB', (100, 100), color='red').save(png_image_path)
# 4. Corrupted file
corrupt_path = 'test_corrupt.jpg'
with open(corrupt_path, 'wb') as f:
    f.write(b"not an image")

def test_analyze(image_path):
    print(f"\n--- Testing {image_path} ---")
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post('http://localhost:5000/api/analyze', files=files)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Verdict: {data.get('verdict')}")
            print(f"Real Prob: {data.get('real_probability'):.4f}")
            print(f"Fake Prob: {data.get('fake_probability'):.4f}")
            print(f"ELA:")
            ela = data.get('ela', {})
            for k, v in ela.items():
                if k == 'visualization':
                    print(f"  {k}: {v[:30]}... (length: {len(v)})")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"Response: {response.text}")

test_analyze(real_image_path)
test_analyze(fake_image_path)
test_analyze(png_image_path)
test_analyze(corrupt_path)
