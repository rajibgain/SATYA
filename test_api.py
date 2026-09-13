import requests
import json

url = "http://127.0.0.1:5000/api/analyze"
file_path = "data/processed/image_ntire/test/fake/0081a9d39a3e5618cd72.jpg"

try:
    with open(file_path, "rb") as f:
        response = requests.post(url, files={"image": f})
    
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
