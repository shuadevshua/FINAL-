import easyocr
import gradio as gr
from PIL import Image
import numpy as np

# Load EasyOCR reader (English only to make it fast)
reader = easyocr.Reader(['en'], gpu=True)

def extract_text(image):
    if image is None:
        return "No image uploaded."

    # Convert to numpy array
    img_array = np.array(image)

    # Perform OCR
    results = reader.readtext(img_array, detail=0)

    # Combine lines
    extracted_text = "\n".join(results)
    return extracted_text if extracted_text else "No text detected."

# Gradio UI
interface = gr.Interface(
    fn=extract_text,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Text Scanner (OCR)",
    description="Upload an image containing text and extract the text using EasyOCR."
)

interface.launch()
