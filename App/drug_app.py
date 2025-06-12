import gradio as gr
import skops.io as sio
import os
import sys

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Improve path handling to find the model file
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(script_dir), "Model", "drug_pipeline.skops")

# Check if model exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    sys.exit(1)

# Get untrusted types and load the model safely
trusted_types = [Pipeline, RandomForestClassifier, np.dtype]
pipe = sio.load(model_path, trusted=trusted_types)


def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features.

    Args:
        age (int): Age of patient
        sex (str): Sex of patient 
        blood_pressure (str): Blood pressure level
        cholesterol (str): Cholesterol level
        na_to_k_ratio (float): Ratio of sodium to potassium in blood

    Returns:
        dict: Prediction probabilities for each drug class
    """
    try:
        features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
        predicted_drug = pipe.predict([features])[0]
        
        # Get prediction probabilities if available
        try:
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba([features])[0]
                classes = pipe.classes_
                result = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
                return result
        except Exception:
            pass
            
        # If probabilities not available, return the prediction as 100% confidence
        return {str(predicted_drug): 1.0}
    except Exception as e:
        return {"Error": str(e)}


# Define inputs with better descriptions
inputs = [
    gr.Slider(15, 74, step=1, label="Age", info="Patient age (15-74 years)"),
    gr.Radio(["M", "F"], label="Sex", info="Patient gender"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure", info="Patient blood pressure level"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol", info="Patient cholesterol level"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K", info="Sodium to Potassium ratio in blood"),
]

# Use Label component to show prediction probabilities
outputs = gr.Label(num_top_classes=5)

# Example cases for users to try
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

title = "Drug Classification"
description = "Enter patient details to predict the most suitable drug type"
article = """
This app predicts the most suitable drug for a patient based on their medical parameters.
Part of the Beginner's Guide to CI/CD for Machine Learning, demonstrating automated 
training, evaluation, and deployment using GitHub Actions.
"""

# Create and launch the interface with additional configuration
demo = gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
    allow_flagging="never",  # Disable flagging
)

if __name__ == "__main__":
    demo.launch(share=False)  # Set share=True to create a public link