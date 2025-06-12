"""
Model evaluation module for drug classification.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance.
    
    Parameters:
    -----------
    model : sklearn estimator
        The trained model to evaluate.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels for X_test.
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "predictions": predictions
    }

def save_metrics(metrics, filepath="Results/metrics.txt"):
    """
    Save evaluation metrics to a file.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics.
    filepath : str, default="Results/metrics.txt"
        Path to save the metrics.
    """
    with open(filepath, "w") as outfile:
        outfile.write(f"\nAccuracy = {metrics['accuracy']}, F1 Score = {metrics['f1_score']}.")

def plot_confusion_matrix(model, y_test, predictions, filepath="Results/model_results.png"):
    """
    Plot and save confusion matrix.
    
    Parameters:
    -----------
    model : sklearn estimator
        The trained model.
    y_test : array-like
        True labels.
    predictions : array-like
        Predicted labels.
    filepath : str, default="Results/model_results.png"
        Path to save the confusion matrix plot.
    """
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.savefig(filepath, dpi=120)