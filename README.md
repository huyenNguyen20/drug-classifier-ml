# MLOps - Drug Classifier

The repository is used for practice and demonstrate how to perform MLOps with AWS. The machine learning model used in the repository is to classify drug types based on patient characteristics.
Kaggle Link for Dataset: https://www.kaggle.com/datasets/prathamtripathi/drug-classification

## Project Structure

```
drug-classifier/
├── Data/               # Data directory
│   └── drug.csv        # Dataset
├── Model/              # Model directory
│   └── drug_pipeline.skops  # Saved model
├── Results/            # Results directory
│   ├── metrics.txt     # Model metrics
│   └── model_results.png  # Confusion matrix
├── src/                # Source code
│   ├── __init__.py
│   ├── data_loader.py  # Data loading functions
│   ├── data_preparation.py  # Data preparation functions
│   ├── model_builder.py  # Model building functions
│   ├── model_evaluation.py  # Model evaluation functions
│   └── model_persistence.py  # Model saving functions
├── notebook.ipynb      # Original notebook
├── requirements.txt    # Project dependencies
└── train.py            # Main training script
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python train.py
```

## Features

- Data loading and preprocessing
- Model training with scikit-learn pipeline
- Model evaluation with accuracy and F1 score
- Confusion matrix visualization
- Model persistence with skops

## Task List
