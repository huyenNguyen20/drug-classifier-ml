"""
Model persistence module for drug classification.
"""
import skops.io as sio

def save_model(model, filepath="Model/drug_pipeline.skops"):
    """
    Save the trained model to disk.
    
    Parameters:
    -----------
    model : sklearn estimator
        The trained model to save.
    filepath : str, default="Model/drug_pipeline.skops"
        Path to save the model.
    """
    sio.dump(model, filepath)