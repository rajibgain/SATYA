"""
SATYA API Server
Wraps the existing inference pipeline and serves the frontend.
"""

import io
import sys
import tempfile
import os
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# Ensure project root is in path for src.image imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image.inference import load_model, analyze_image
from src.image.metadata import extract_metadata
import torch

# ── Configuration ──
MODEL_PATH = 'models/image_ntire_unfrozen/best_image_model.pth'
THRESHOLD = 0.50  # Production threshold (validated on held-out test set)
FRONTEND_DIR = Path(__file__).parent / 'frontend'

# ── App setup ──
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
CORS(app)

# ── Load model once at startup ──
print('[SATYA] Loading model...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(MODEL_PATH, device)
print(f'[SATYA] Model loaded on {device}')


# ── Routes ──

@app.route('/')
def index():
    """Serve the frontend SPA."""
    return send_from_directory(str(FRONTEND_DIR), 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze an uploaded image.
    Expects multipart form with 'image' field.
    Returns JSON with verdict and probabilities.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read the image into PIL
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Extract metadata
        metadata = extract_metadata(image_bytes)

        # Error Level Analysis
        from src.image.ela import analyze_ela
        ela = analyze_ela(image_bytes, quality=90)
        
        # Image Integrity
        from src.image.integrity import analyze_image_integrity
        integrity = analyze_image_integrity(image_bytes)

        # Save to a temporary path for analyze_image (which expects a path)
        # Instead, we'll inline the logic to avoid temp files
        from src.image.preprocess import preprocess_image
        input_tensor = preprocess_image(image).to(device)

        # Image Artifacts
        from src.image.artifacts import analyze_image_artifacts
        artifacts = analyze_image_artifacts(image_bytes)

        # AI-Generated Image Detection (Feature 5)
        from src.ai_detection.inference import analyze_ai_generation
        ai_generated = analyze_ai_generation(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)

        real_prob = probs[0].item()
        fake_prob = probs[1].item()

        verdict = 'LIKELY FAKE' if fake_prob >= THRESHOLD else 'LIKELY REAL'

        print(f"Artifacts output keys: {artifacts.keys() if artifacts else artifacts}", flush=True)

        return jsonify({
            'verdict': verdict,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'threshold': THRESHOLD,
            'model_name': 'EfficientNet-B0',
            'metadata': metadata,
            'ela': ela,
            'integrity': integrity,
            'artifacts': artifacts,
            'ai_generated': ai_generated
        })

    except UnidentifiedImageError:
        return jsonify({'error': 'Invalid or unsupported image file.'}), 400
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': 'EfficientNet-B0', 'device': str(device)})


@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    """
    Analyze an uploaded video.
    Expects multipart form with 'video' field.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        from src.video.inference import infer_video
        
        # Save video to a temporary file for cv2
        fd, temp_path = tempfile.mkstemp(suffix=Path(file.filename).suffix)
        os.close(fd)
        file.save(temp_path)
        
        try:
            result = infer_video(temp_path, checkpoint_path=MODEL_PATH, device=device)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify(result)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """
    Analyze an uploaded text file.
    Expects multipart form with 'text' field.
    """
    if 'text' not in request.files:
        return jsonify({'error': 'No text file provided'}), 400

    file = request.files['text']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        from src.text.inference import infer_text
        
        # Save text to a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        file.save(temp_path)
        
        try:
            result = infer_text(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify(result)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'traceback': tb}), 500


if __name__ == '__main__':
    print('[SATYA] Starting server at http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
