import gradio as gr
from bert_predict import predict_comment
import pandas as pd

# ---------------------------------
# ✅ Wrapper: add rating + bar plot
# ---------------------------------
def classify_comment(comment, threshold=0.3):
    """Return toxicity rating + DataFrame for Gradio BarPlot."""
    results = predict_comment(comment)

    # Apply threshold: anything below the threshold is set to 0
    results = {
        label: (score if score >= threshold else 0.0)
        for label, score in results.items()
    }

    # -------------------------------
    # 🔹 Determine toxicity rating
    # -------------------------------
    # You can base this on the 'toxic' label
    toxic_score = results.get("toxic", 0.0)

    if toxic_score < 0.3:
        toxic_class = "🟢 Clean / Non-toxic"
    elif toxic_score < 0.6:
        toxic_class = "🟠 Mildly Toxic"
    else:
        toxic_class = "🔴 Highly Toxic"

    # -------------------------------
    # 🔹 Convert to DataFrame for plot
    # -------------------------------
    df = pd.DataFrame({
        "label": list(results.keys()),
        "score": list(results.values())
    })

    # Return both the textual class and the bar data
    return toxic_class, df


# -------------------------------
# ✅ Build Gradio Interface
# -------------------------------
demo = gr.Interface(
    fn=classify_comment,
    inputs=gr.Textbox(
        label="Enter a comment",
        placeholder="Type something toxic or polite...",
        lines=3
    ),
    outputs=[
        gr.Textbox(label="Toxicity Rating"),  # 🟢 add class text output
        gr.BarPlot(
            label="Toxicity Breakdown",
            x="label",
            y="score",
            color="label",
            title="Toxicity Levels per Category"
        )
    ],
    title="Toxic Comment Classifier (BERT Multi-Label)",
    description=(
        "This BERT model detects multiple types of toxicity — such as insults, "
        "threats, or hate speech — and shows both a toxicity rating and per-category intensity."
    ),
    examples=[
        ["You are such an idiot!"],
        ["I hope you have a great day!"],
        ["I'll find you and make you pay."],
        ["You don't belong here because of your race."],
        ["This is disgusting behavior!"]
    ]
)

# -------------------------------
# ✅ Launch app (debug=True)
# -------------------------------
if __name__ == "__main__":
    demo.launch(debug=True)
