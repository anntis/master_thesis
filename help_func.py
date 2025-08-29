import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor

def convert_to_long_format(df, id_vars, var_name, value_name):
    """
    Converts a DataFrame from wide format to long format.

    Parameters:
        df (pd.DataFrame): The source DataFrame in wide format.
        id_vars (list): Columns to remain unchanged.
        var_name (str): Name of the column for variable names.
        value_name (str): Name of the column for cell values.

    Returns:
        pd.DataFrame: A DataFrame in long format with the 'Year' type converted, if applicable.
    """
    # Use melt to convert to long format
    df_long = df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)
    
    # Convert 'Year' to numeric format (if 'Year' is used as var_name)
    if var_name == 'Year':
        df_long[var_name] = df_long[var_name].astype(int)
    
    return df_long



def save_dataframe_as_image(df, filename):
    """
    Saves a DataFrame as a PNG image.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): Path to the file, including the name and .png extension.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    plt.savefig(filename)
    

def find_optimal_learners(X, y, is_classifier=False):
    """
    Finds optimal hyperparameters for a RandomForest model using GridSearchCV.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        is_classifier (bool): True if the model is a classifier, False for a regressor.

    Returns:
        A fitted GridSearchCV object.
    """
    if is_classifier:
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [30, 35, 50, 100, 200],
            'max_depth': [3, 5, 10],
            'min_samples_leaf': [1, 3, 5]
        }
    else:
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [30, 35, 50, 100, 200],
            'max_depth': [3, 5, 10],
            'min_samples_leaf': [1, 3, 5]
        }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error' if not is_classifier else 'accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    
    print(f"\n--- Optimal Parameters for {'Classifier' if is_classifier else 'Regressor'} ---")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def find_optimal_lasso(X, y, is_classifier=False):
    """
    Finds the optimal alpha for Lasso models using GridSearchCV.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        is_classifier (bool): True if the model is a classifier, False for a regressor.

    Returns:
        A fitted GridSearchCV object.
    """
    if is_classifier:
        model = LogisticRegression(solver='liblinear', random_state=42)
        # For Logistic Regression, the regularization strength is controlled by 'C' (inverse of alpha).
        # Penalty as 'l1' for Lasso regularization.
        param_grid = {'C': np.logspace(-4, 0, 100), 'penalty': ['l1']}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    else:
        model = Lasso(random_state=42)
        # For Lasso, the regularization strength is controlled by 'alpha'.
        param_grid = {'alpha': np.logspace(-4, 0, 100)}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

    grid_search.fit(X, y)
    
    print(f"\n--- Optimal Parameters for Lasso {'Classifier' if is_classifier else 'Regressor'} ---")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_


def find_optimal_xgboost(X, y, is_classifier=False):
    """
    Finds optimal hyperparameters for an XGBoost model using GridSearchCV.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        is_classifier (bool): True if the model is a classifier, False for a regressor.

    Returns:
        A fitted GridSearchCV object.
    """
    if is_classifier:
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
        param_grid = {
            'n_estimators': [30, 35, 50, 100, 200],
            'max_depth': [1, 3, 5, 10],
            'learning_rate': np.linspace(0.01, 0.2, 10)
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    else:
        model = XGBRegressor(objective="reg:squarederror", random_state=42)
        param_grid = {
            'n_estimators': [30, 35, 50, 100, 200],
            'max_depth': [1, 3, 5, 10],
            'learning_rate': np.linspace(0.01, 0.2, 10)
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    
    grid_search.fit(X, y)
    
    print(f"\n--- Optimal Parameters for XGBoost {'Classifier' if is_classifier else 'Regressor'} ---")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_