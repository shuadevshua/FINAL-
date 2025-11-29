from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from bert_predict import predict_comment
import os
import io
import numpy as np
from PIL import Image
import easyocr
import torch
import json
from datetime import datetime
import threading

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Enable CORS for frontend

# History storage
HISTORY_FILE = 'analysis_history.json'
MAX_HISTORY_ENTRIES = 1000
history_lock = threading.Lock()

# Initialize EasyOCR reader (lazy loading)
ocr_reader = None

def get_ocr_reader():
    """Lazy load OCR reader to avoid loading on startup."""
    global ocr_reader
    if ocr_reader is None:
        print("[INFO] Initializing EasyOCR reader...")
        try:
            use_gpu = torch.cuda.is_available()
            ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
            print(f"[INFO] EasyOCR reader initialized (GPU: {use_gpu}).")
        except Exception as e:
            print(f"[WARNING] GPU initialization failed, using CPU: {e}")
            ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("[INFO] EasyOCR reader initialized (CPU).")
    return ocr_reader

def enhance_image(image):
    """Enhance image for better OCR accuracy."""
    try:
        from PIL import ImageEnhance
        
        # Ensure it's a PIL Image
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
        
        # Enhance contrast for better text visibility
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        # Convert back to numpy array
        img_array = np.array(pil_img)
        return img_array
    except Exception as e:
        print(f"[WARNING] Image enhancement failed, using original: {e}")
        # Return original as numpy array
        if isinstance(image, Image.Image):
            return np.array(image)
        return image

def extract_text_from_image(image_file):
    """Extract text from uploaded image using EasyOCR with image enhancement."""
    try:
        # Read image from file
        image_file.seek(0)  # Reset file pointer
        image = Image.open(io.BytesIO(image_file.read()))
        image = image.convert('RGB')
        
        # Enhance image for better OCR
        enhanced_array = enhance_image(image)
        
        # Extract text using EasyOCR
        reader = get_ocr_reader()
        results = reader.readtext(enhanced_array, detail=0, paragraph=True)
        
        # Combine all detected text lines
        extracted_text = "\n".join(results).strip()
        
        print(f"[INFO] OCR extracted {len(extracted_text)} characters from image")
        
        return extracted_text
    except Exception as e:
        print(f"[ERROR] OCR extraction failed: {e}")
        raise

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for toxicity prediction."""
    try:
        data = request.get_json()
        comment = data.get('comment', '')
        
        if not comment:
            return jsonify({'error': 'Comment is required'}), 400
        
        # Get predictions (raw scores, no threshold filtering)
        results = predict_comment(comment)
        
        # Determine toxicity rating based on raw toxic score
        toxic_score = results.get("toxic", 0.0)
        
        if toxic_score < 0.3:
            toxic_class = "Clean / Non-toxic"
            severity = "low"
            emoji = "🟢"
        elif toxic_score < 0.6:
            toxic_class = "Mildly Toxic"
            severity = "medium"
            emoji = "🟠"
        else:
            toxic_class = "Highly Toxic"
            severity = "high"
            emoji = "🔴"
        
        return jsonify({
            'success': True,
            'rating': toxic_class,
            'emoji': emoji,
            'severity': severity,
            'scores': results  # Return raw scores without filtering
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocr', methods=['POST'])
def ocr():
    """API endpoint for OCR text extraction from images."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Extract text from image
        extracted_text = extract_text_from_image(image_file)
        
        if not extracted_text:
            return jsonify({
                'success': True,
                'text': '',
                'message': 'No text detected in the image.'
            })
        
        return jsonify({
            'success': True,
            'text': extracted_text
        })
    
    except Exception as e:
        print(f"[ERROR] OCR endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocr_predict', methods=['POST'])
def ocr_predict():
    """API endpoint: Extract text from image + run toxicity analysis."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Extract text from image
        extracted_text = extract_text_from_image(image_file)
        
        if not extracted_text:
            return jsonify({
                'success': True,
                'text': '',
                'message': 'No text detected in the image.',
                'rating': 'No text',
                'emoji': '⚪',
                'severity': 'none',
                'scores': {}
            })
        
        # Run toxicity prediction on extracted text
        results = predict_comment(extracted_text)
        
        # Determine toxicity rating based on raw toxic score
        toxic_score = results.get("toxic", 0.0)
        
        if toxic_score < 0.3:
            toxic_class = "Clean / Non-toxic"
            severity = "low"
            emoji = "🟢"
        elif toxic_score < 0.6:
            toxic_class = "Mildly Toxic"
            severity = "medium"
            emoji = "🟠"
        else:
            toxic_class = "Highly Toxic"
            severity = "high"
            emoji = "🔴"
        
        return jsonify({
            'success': True,
            'text': extracted_text,
            'rating': toxic_class,
            'emoji': emoji,
            'severity': severity,
            'scores': results
        })
    
    except Exception as e:
        print(f"[ERROR] OCR predict endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

def load_history_from_file():
    """Load history from JSON file."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                return history
        return []
    except Exception as e:
        print(f"[WARNING] Failed to load history from file: {e}")
        return []

def save_history_to_file(history):
    """Save history to JSON file."""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(history)} history entries to file")
    except Exception as e:
        print(f"[WARNING] Failed to save history to file: {e}")

# Load history on startup
analysis_history = load_history_from_file()

@app.route('/api/history/save', methods=['POST'])
def save_history():
    """API endpoint to save analysis history."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        entry = {
            'id': data.get('id', int(datetime.now().timestamp() * 1000)),
            'text': data.get('text', ''),
            'fullText': data.get('fullText', data.get('text', '')),
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'rating': data.get('rating', 'Unknown'),
            'severity': data.get('severity', 'low'),
            'scores': data.get('scores', {})
        }
        
        with history_lock:
            # Add to beginning (most recent first)
            analysis_history.insert(0, entry)
            
            # Keep only last MAX_HISTORY_ENTRIES entries
            if len(analysis_history) > MAX_HISTORY_ENTRIES:
                analysis_history[:] = analysis_history[:MAX_HISTORY_ENTRIES]
            
            # Save to file
            save_history_to_file(analysis_history)
        
        return jsonify({
            'success': True,
            'message': 'History saved successfully',
            'total_entries': len(analysis_history)
        })
    
    except Exception as e:
        print(f"[ERROR] Save history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/get', methods=['GET'])
def get_history():
    """API endpoint to get analysis history."""
    try:
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', 0, type=int)
        
        with history_lock:
            history = analysis_history.copy()
        
        # Apply pagination if requested
        if limit:
            history = history[offset:offset + limit]
        
        # Calculate statistics
        total = len(analysis_history)
        safe = len([e for e in analysis_history if e.get('severity') == 'low'])
        toxic = total - safe
        safe_percent = round((safe / total * 100)) if total > 0 else 0
        toxic_percent = round((toxic / total * 100)) if total > 0 else 0
        
        return jsonify({
            'success': True,
            'history': history,
            'stats': {
                'total': total,
                'safe': safe,
                'toxic': toxic,
                'safePercent': safe_percent,
                'toxicPercent': toxic_percent
            },
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': len(analysis_history)
            }
        })
    
    except Exception as e:
        print(f"[ERROR] Get history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """API endpoint to clear all analysis history."""
    try:
        with history_lock:
            analysis_history.clear()
            save_history_to_file(analysis_history)
        
        return jsonify({
            'success': True,
            'message': 'History cleared successfully'
        })
    
    except Exception as e:
        print(f"[ERROR] Clear history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    print("🚀 Starting Flask API server...")
    print("📡 API will be available at http://localhost:5000")
    print("🌐 Frontend available at http://localhost:5000/")
    print("🔌 API endpoints:")
    print("   - POST /api/predict - Text toxicity detection")
    print("   - POST /api/ocr - Extract text from image")
    print("   - POST /api/ocr_predict - OCR + toxicity detection")
    print("   - GET /api/health - Health check")
    print("   - POST /api/history/save - Save analysis history")
    print("   - GET /api/history/get - Get analysis history")
    print("   - POST /api/history/clear - Clear analysis history")
    app.run(debug=True, port=5000)

