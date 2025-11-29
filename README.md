# MLNLPSoftEng - Neural Toxicity Analyzer

BERT-based toxic comment classifier with OCR capabilities, a futuristic 3D web interface **and** a Chrome extension built with React.

## 🚀 Quick Start (Flask + Web UI)

1. **Install Dependencies** (if needed): 
   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Start Server**: Double-click `start_server.bat` (or run `python app.py` inside venv)
3. **Open Browser**: Go to http://localhost:5000/

## 📁 Project Structure

- **app.py** – Unified Flask API backend with OCR + toxicity detection
- **bert_predict.py** – Loads the trained BERT model for inference
- **frontend/** – Light-themed web UI with text input and image upload
- **chrome-extension/** – React + Vite Chrome extension
- **bert_toxic_model_multilabel_final/** – Trained BERT model files
- **venv/** – **Single** Python virtual environment used by all scripts

## 🔧 Virtual Environment

- Everything runs in **venv**
- Activate manually: `.\venv\Scripts\Activate.ps1` (PowerShell)
- Start scripts via `start_server.bat` / `start_server.ps1` for convenience

## 📚 Additional Docs

- chrome-extension/README.md – Build & installation steps for the React MV3 extension

## 🎯 API Endpoints

- **POST /api/predict** – Text toxicity detection (JSON: `{"comment": "..."}`)
- **POST /api/ocr** – Extract text from uploaded image (multipart/form-data: `image`)
- **POST /api/ocr_predict** – OCR + toxicity detection in one call (multipart/form-data: `image`)
- **GET /api/health** – Health check endpoint

## 🎨 Features

### Web UI Features:
- **Text Input Mode**: Direct text toxicity analysis
- **Image Upload Mode**: Upload images with text, extract with OCR, then analyze toxicity
- 3D particle background + interactive charts
- Real-time toxicity scoring for 6 categories:
  - Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate

### Technical Features:
- **EasyOCR Integration**: Automatic text extraction from images
- **BERT Multi-Label Classification**: Advanced neural network toxicity detection
- **Lazy Loading**: OCR reader loads on first use (faster startup)
- **GPU Support**: Automatic GPU detection for faster OCR processing
- Works completely offline (models + API run locally)

## 💡 Usage Examples

### Text Analysis:
1. Click "💬 Text Input" tab
2. Type or paste text
3. Click "Analyze"

### Image Analysis:
1. Click "📷 Image Upload" tab
2. Upload an image (drag & drop or click to browse)
3. Wait for OCR to extract text (shown automatically)
4. Click "Analyze" to check toxicity of extracted text
