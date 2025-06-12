"""
Data loading module for drug classification.
"""
import pandas as pd

def load_data(filepath="Data/drug.csv", shuffle=True):
    """
    Load the drug dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str, default="Data/drug.csv"
        Path to the CSV file containing the drug data.
    shuffle : bool, default=True
        Whether to shuffle the data.
        
    Returns:
    --------
    pandas.DataFrame
        The loaded drug dataset.
    """
    drug_df = pd.read_csv(filepath)
    if shuffle:
        drug_df = drug_df.sample(frac=1)
    return drug_df