import torch
import easyocr
from PIL import Image
import numpy as np
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import re
import pyperclip  # for clipboard access

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "./bert_toxic_model_multilabel_final"
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
THRESHOLD = 0.5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ----------------------------
# LOAD BERT
# ----------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ----------------------------
# INIT OCR
# ----------------------------
LANGUAGES = ['en', 'es', 'fr']
ocr_reader = easyocr.Reader(LANGUAGES, gpu=torch.cuda.is_available())

# ----------------------------
# EXTRACT TEXT FROM IMAGE
# ----------------------------
def extract_text_from_image(pil_image):
    """Minimal preprocessing OCR, removes cursor/artifacts"""
    image = pil_image.convert("RGB")
    img_array = np.array(image)

    results = ocr_reader.readtext(img_array, detail=1)
    print("OCR Raw Results:", results)  # debug

    texts = [res[1] for res in results]
    detected_text = " ".join(texts).strip()

    detected_text = re.sub(r'[\|\_\.]+$', '', detected_text).strip()
    return detected_text

# ----------------------------
# BERT PREDICTION
# ----------------------------
def predict_text(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy().tolist()

    labels_and_scores = [
        {"label": lbl, "score": float(prob), "pred": bool(prob >= THRESHOLD)}
        for lbl, prob in zip(LABELS, probs)
    ]
    positives = [item["label"] for item in labels_and_scores if item["pred"]]
    return labels_and_scores, positives

# ----------------------------
# CLASSIFY FUNCTION
# ----------------------------
def classify(text_input, image_input):
    if image_input is not None:
        used_text = extract_text_from_image(image_input) or text_input
    else:
        used_text = text_input

    if not used_text.strip():
        return "No text detected.", "", []

    labels_and_scores, positives = predict_text(used_text)
    display_lines = [
        f"{item['label']}: {item['score']:.4f} -> {'YES' if item['pred'] else 'no'}"
        for item in labels_and_scores
    ]
    display_text = "\n".join(display_lines)
    return used_text, display_text, positives

# ----------------------------
# PASTE FROM CLIPBOARD FUNCTION
# ----------------------------
def paste_clipboard():
    try:
        return pyperclip.paste()
    except Exception:
        return ""

# ----------------------------
# GRADIO UI
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# BERT Toxicity Detector + OCR (Clipboard Ready)")
    gr.Markdown(
        "Paste text from clipboard, type manually, or upload an image. "
        "OCR detects text (minimal preprocessing) and sends it to BERT for classification."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Type or Paste Text Here",
                placeholder="Enter text or paste from clipboard...",
                lines=6,
                interactive=True,
                autofocus=True
            )
            paste_btn = gr.Button("Paste from Clipboard")
            image_input = gr.Image(label="Upload image", type="pil")
            run_btn = gr.Button("Classify")
        with gr.Column(scale=1):
            used_text_output = gr.Textbox(label="Processed Text (OCR)", interactive=False, lines=6)
            results_output = gr.Textbox(label="Label scores & predictions", interactive=False, lines=8)
            positives_output = gr.Textbox(label="Positive Labels", interactive=False, lines=2)

    # Button actions
    paste_btn.click(fn=paste_clipboard, inputs=[], outputs=text_input)
    run_btn.click(fn=classify, inputs=[text_input, image_input],
                  outputs=[used_text_output, results_output, positives_output])

print("🚀 BERT OCR app running! Open http://localhost:7860 in your browser.")
demo.launch(server_name="0.0.0.0", server_port=7860)
