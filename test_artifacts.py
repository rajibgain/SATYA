import requests
import json
import sys

URL = 'http://localhost:5000/api/analyze'

def test_image(path):
    print(f"\n--- Testing {path} ---")
    with open(path, 'rb') as f:
        files = {'image': (path, f, 'image/jpeg')}
        response = requests.post(URL, files=files)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
        
    data = response.json()
    
    print(f"Verdict: {data.get('verdict')}")
    print(f"Real Prob: {data.get('real_probability'):.4f}")
    print(f"Fake Prob: {data.get('fake_probability'):.4f}")
    
    arts = data.get('artifacts', {})
    if not arts.get('available'):
        print(f"Artifacts object: {arts}")
        print(f"Artifacts NOT available. Error: {arts.get('error')}")
    else:
        print("Artifacts generated successfully.")
        print(f"Noise available: {arts['noise_residual']['available']}")
        print(f"Local Var available: {arts['local_variation']['available']}")
        print(f"Edge available: {arts['edge_analysis']['available']}")
        print(f"JPEG analysis available: {arts['jpeg_block_analysis']['available']}")
        print(f"FFT available: {arts['frequency_analysis']['available']}")

if __name__ == '__main__':
    # Test a fake image
    test_image('data/processed/image_ntire/test/fake/0081a9d39a3e5618cd72.jpg')
    # Test a real image
    test_image('data/processed/image_ntire/test/real/0016ed96770dc80ff1b5.jpg')
