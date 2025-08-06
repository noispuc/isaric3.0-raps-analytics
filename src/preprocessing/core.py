import pandas as pd
import numpy as np
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from .utils import calc_median_iqr, is_binary

def run_mice_imputation(df, interest_vars, aux_vars, max_iter=10, random_state=100, prefix_sep='!'):
    """
    Performs MICE (Multiple Imputation by Chained Equations) on selected columns.

    Converts categorical and binary variables appropriately before imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset to be imputed.
    interest_vars : list of str
        Variables of interest that need to be imputed.
    aux_vars : list of str
        Auxiliary variables used to support the imputation process.
    max_iter : int, optional
        Maximum number of imputation iterations (default is 10).
    random_state : int, optional
        Random seed for reproducibility (default is 100).
    prefix_sep : str, optional
        Separator used when creating dummy variables from categoricals (default is '!').

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values, including both interest and auxiliary variables.
    """
    
    def pre_imputation(df, selected_vars):
        df = df[selected_vars].copy()
        for var in df.columns:
            if df[var].dtype == 'object' and is_binary(df[var]):
                df[var] = df[var].astype(bool)
            elif df[var].dtype == 'object':
                df = pd.get_dummies(df, columns=[var], prefix_sep=prefix_sep, dtype=bool)
        return df

    selected_vars = list(set(interest_vars + aux_vars))
    df_copy = pre_imputation(df.copy(), selected_vars)
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputed_array = imputer.fit_transform(df_copy)
    return pd.DataFrame(imputed_array, columns=df_copy.columns)

def create_mnar(df, interest_col, aux_cols, missing_pct, reps=30):
    """
    Generates datasets with MNAR (Missing Not At Random) values.

    Introduces missing values in `interest_col` based on the upper quartile of `aux_cols`.

    Parameters
    ----------
    df : pd.DataFrame
        Original clean dataset.
    interest_col : str
        Target column to apply MNAR mask.
    aux_cols : list of str
        Variables used to determine MNAR pattern (values above Q3).
    missing_pct : float
        Percentage of total rows to be set as missing (between 0 and 1).
    reps : int, optional
        Number of dataset replications to generate (default is 30).

    Returns
    -------
    tuple
        A pair containing:
        - List of pd.DataFrame with MNAR missingness applied.
        - pd.DataFrame of original cleaned data with no missing values.
    """
    df = df.dropna(subset=[interest_col] + aux_cols).reset_index(drop=True)
    q3s = {col: calc_median_iqr(df[col])[2] for col in aux_cols}
    total_rows = len(df)
    n_missing = int(total_rows * missing_pct)
    datasets = []

    for _ in range(reps):
        idxs = set()
        while len(idxs) < n_missing:
            for col in aux_cols:
                q3 = q3s[col]
                sample = df[df[col] > q3]
                idxs.update(sample.sample(min(n_missing - len(idxs), sample.shape[0]), random_state=random.randint(0, 1000)).index)
        sampled_idxs = random.sample(list(idxs), n_missing)
        df_missing = df.copy()
        df_missing.loc[sampled_idxs, interest_col] = np.nan
        datasets.append(df_missing)
    return datasets, df

def evaluate_imputation(original_df, datasets, interest_col, aux_cols):
    """
    Evaluates MICE imputation using MAE and MAPE metrics across replications.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original dataset without missing values.
    datasets : list of pd.DataFrame
        Replicated datasets with MNAR values to be imputed.
    interest_col : str
        Column being evaluated for imputation accuracy.
    aux_cols : list of str
        Variables supporting the imputation process.

    Returns
    -------
    list of dict
        A list where each dict contains:
        - 'MAE': Mean Absolute Error
        - 'MAPE': Mean Absolute Percentage Error
    """
    stats = []

    for df in datasets:
        missing_idxs = df[df[interest_col].isna()].index
        imputed_df = run_mice_imputation(df, interest_vars=[interest_col], aux_vars=aux_cols)

        abs_errors = []
        mapes = []

        for i in missing_idxs:
            original_val = original_df.loc[i, interest_col]
            imputed_val = imputed_df.loc[i, interest_col]
            abs_error = abs(original_val - imputed_val)
            abs_errors.append(abs_error)
            if original_val != 0:
                mapes.append(abs_error / abs(original_val))

        mae = np.mean(abs_errors)
        mape = np.mean(mapes) if mapes else np.nan
        stats.append({'MAE': mae, 'MAPE': mape})

    return stats
