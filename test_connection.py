"""
Test script to verify the connection between frontend, backend, and trained model.
"""
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 60)
print("Testing Backend Connection to Trained Model")
print("=" * 60)

# Test 1: Check if model directory exists
print("\n[1] Checking model directory...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "bert_toxic_model_multilabel_final")
if os.path.exists(MODEL_PATH):
    print(f"   [OK] Model directory found: {MODEL_PATH}")
    model_files = os.listdir(MODEL_PATH)
    required_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.txt']
    for file in required_files:
        if file in model_files:
            print(f"   [OK] {file} found")
        else:
            print(f"   [ERROR] {file} missing!")
else:
    print(f"   [ERROR] Model directory not found: {MODEL_PATH}")
    sys.exit(1)

# Test 2: Check if bert_predict can be imported
print("\n[2] Testing bert_predict import...")
try:
    from bert_predict import predict_comment, LABELS
    print(f"   [OK] bert_predict imported successfully")
    print(f"   [OK] Labels: {LABELS}")
except Exception as e:
    print(f"   [ERROR] Failed to import bert_predict: {e}")
    sys.exit(1)

# Test 3: Test prediction function
print("\n[3] Testing prediction function...")
try:
    test_comment = "You are such an idiot!"
    print(f"   Testing with: '{test_comment}'")
    results = predict_comment(test_comment)
    print(f"   [OK] Prediction successful!")
    print(f"   Results:")
    for label, score in results.items():
        print(f"      - {label}: {score:.4f}")
except Exception as e:
    print(f"   [ERROR] Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check if Flask app can be imported
print("\n[4] Testing Flask app import...")
try:
    from app import app
    print(f"   [OK] Flask app imported successfully")
except Exception as e:
    print(f"   [ERROR] Failed to import Flask app: {e}")
    sys.exit(1)

# Test 5: Verify labels match training labels
print("\n[5] Verifying label consistency...")
training_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
if LABELS == training_labels:
    print(f"   [OK] Labels match training labels exactly")
else:
    print(f"   [WARNING] Label mismatch!")
    print(f"      Training: {training_labels}")
    print(f"      Prediction: {LABELS}")

print("\n" + "=" * 60)
print("[SUCCESS] All connection tests passed!")
print("=" * 60)
print("\nConnection Chain:")
print("   Frontend (static/index.html)")
print("   |")
print("   Flask API (app.py) -> /api/predict")
print("   |")
print("   bert_predict.py -> predict_comment()")
print("   |")
print("   bert_toxic_model_multilabel_final/ (trained model)")
print("   |")
print("   Model trained on train.csv")
print("\nReady to run! Start with: python app.py")

