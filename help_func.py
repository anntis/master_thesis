import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
    