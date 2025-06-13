"""
Main training script for drug classification model.
"""

from src.data_loader import load_data
from src.data_preparation import prepare_data
from src.model_builder import create_pipeline
from src.model_evaluation import evaluate_model, save_metrics, plot_confusion_matrix
from src.model_persistence import save_model


def main():
    """
    Main function to train and evaluate the drug classification model.
    """
    # Load data
    print("Loading data...")
    drug_df = load_data()

    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(drug_df)

    # Create and train model
    print("Training model...")
    pipe = create_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(pipe, X_test, y_test)

    # Print results
    print(f"Accuracy: {metrics['accuracy']:.2%} F1: {metrics['f1_score']:.2f}")

    # Save metrics
    save_metrics(metrics)

    # Plot confusion matrix
    plot_confusion_matrix(pipe, y_test, metrics["predictions"])

    # Save model
    print("Saving model...")
    save_model(pipe)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
