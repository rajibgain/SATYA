import requests
import json
import sys

def test_image(filepath):
    print(f"\n--- Testing {filepath} ---")
    try:
        with open(filepath, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://127.0.0.1:5000/api/analyze', files=files)
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Verdict: {data['verdict']}")
            print(f"Real Prob: {data['real_probability']:.4f}")
            print(f"Fake Prob: {data['fake_probability']:.4f}")
            
            integrity = data.get('integrity', {})
            print("Integrity:")
            for k, v in integrity.items():
                if k == 'jpeg':
                    print(f"  jpeg:")
                    for jk, jv in v.items():
                        print(f"    {jk}: {jv}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # 1. Known NTIRE real JPEG
    test_image('data/processed/image_ntire/test/real/0016ed96770dc80ff1b5.jpg')
    # 2. Known NTIRE fake JPEG
    test_image('data/processed/image_ntire/test/fake/0081a9d39a3e5618cd72.jpg')
    # 3. PNG
    test_image('test_normal.png')
    # 4. Corrupt image
    test_image('test_corrupt.jpg')
