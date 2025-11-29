# install_dependencies.py

import subprocess
import sys
import platform
import os

def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# -----------------------------
# 1️⃣ Core libraries
# -----------------------------
install("numpy")
install("pandas")
install("matplotlib")
install("seaborn")
install("tqdm")
install("scikit-learn")

# -----------------------------
# 2️⃣ Hugging Face Transformers
# -----------------------------
install("transformers[sentencepiece]")  # includes tokenizers

# -----------------------------
# 3️⃣ Detect GPU and install PyTorch accordingly
# -----------------------------
print("\n[INFO] Detecting GPU for PyTorch installation...")

try:
    import torch
    print("[INFO] PyTorch is already installed.")
except ImportError:
    has_cuda = False
    try:
        # Check NVIDIA GPU
        import subprocess
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            has_cuda = True
    except Exception:
        has_cuda = False

    if has_cuda:
        print("[INFO] NVIDIA GPU detected! Installing PyTorch with CUDA support...")
        # You can customize the CUDA version depending on your GPU & OS
        # This example uses CUDA 12.1 (RTX 4060 compatible)
        install("torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121")
        install("torchvision==0.17.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
        install("torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("[INFO] No NVIDIA GPU detected. Installing CPU-only PyTorch...")
        install("torch")
        install("torchvision")
        install("torchaudio")

# -----------------------------
# 4️⃣ Gradio for deployment
# -----------------------------
install("gradio")

print("\n✅ All dependencies installed successfully!")
