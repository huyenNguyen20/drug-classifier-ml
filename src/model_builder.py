"""
Model building module for drug classification.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def create_pipeline(random_state=125):
    """
    Create a machine learning pipeline for drug classification.
    
    Parameters:
    -----------
    random_state : int, default=125
        Random state for reproducibility.
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        The machine learning pipeline.
    """
    # Define column indices
    cat_col = [1, 2, 3]  # Sex, BP, Cholesterol
    num_col = [0, 4]     # Age, Na_to_K
    
    # Create preprocessing transformer
    transform = ColumnTransformer(
        [
            ("encoder", OrdinalEncoder(), cat_col),
            ("num_imputer", SimpleImputer(strategy="median"), num_col),
            ("num_scaler", StandardScaler(), num_col),
        ]
    )
    
    # Create pipeline
    pipe = Pipeline(
        steps=[
            ("preprocessing", transform),
            ("model", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ]
    )
    
    return pipe