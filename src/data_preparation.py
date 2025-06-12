"""
Data preparation module for drug classification.
"""
from sklearn.model_selection import train_test_split

def prepare_data(df, target_col="Drug", test_size=0.3, random_state=125):
    """
    Split the data into training and testing sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the drug data.
    target_col : str, default="Drug"
        The name of the target column.
    test_size : float, default=0.3
        The proportion of the dataset to include in the test split.
    random_state : int, default=125
        Controls the shuffling applied to the data before applying the split.
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test - The split data.
    """
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test