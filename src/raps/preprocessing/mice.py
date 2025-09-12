"""
MICE Imputation Orchestrator.

Este módulo realiza simulações de valores ausentes MNAR em dados clínicos 
e avalia a performance da imputação usando o método MICE. Para cada 
variável de interesse e diferentes níveis de porcentagem de ausência, 
o script gera gráficos e estatísticas descritivas.

Uso recomendado:
---------------
Executar como script principal:
    python mice.py

Ou importar e chamar a função `main()` diretamente.

Dependências:
-------------
- pandas
- numpy
- seaborn
- matplotlib
- sklearn
- módulo interno: core.py
- módulo interno: utils.py
"""

import pandas as pd
import numpy as np
from .core import create_mnar, evaluate_imputation
from .utils import extract_stat

def evaluate_mice_mnar(df,cols_to_test,missing_levels):
    """
    Evaluate MNAR imputation performance across variables and missingness levels.

    This function simulates Missing-Not-At-Random (MNAR) patterns for each target
    column, generates imputed datasets via `create_mnar`, evaluates the imputation
    quality with `evaluate_imputation`, extracts statistics (MAE and MAPE) using
    `extract_stat`, and returns an aggregated summary table (mean and std) per
    (column, missing_pct).

    Args:
        df (pd.DataFrame): The original dataset.
        cols_to_test (Sequence[str]): Columns to be individually subjected to MNAR simulation.
        missing_levels (Sequence[float]): Fractions of MNAR missingness to test (each in [0, 1]).
       

    Returns:
        pd.DataFrame: Summary table with one row per (column, missing_pct) and columns:
            - "column": Target column evaluated.
            - "missing_pct": MNAR level used (float in [0, 1]).
            - "MAE_mean": Mean MAE across imputation runs.
            - "MAE_std":  Standard deviation of MAE.
            - "MAPE_mean": Mean MAPE across imputation runs.
            - "MAPE_std":  Standard deviation of MAPE.

    Raises:
        KeyError: If any column in `cols_to_test` is not present in `df`.
        ValueError: If `missing_levels` contains values outside [0, 1] or is empty.

    Example:
            summary = mice(df, cols_to_test=["age", "crp"], missing_levels=[0.1, 0.3, 0.5])
            summary.sort_values(["column", "missing_pct"]).head()
    """

    results = []

    for interest_col in cols_to_test:
        aux_cols = [c for c in cols_to_test if c != interest_col]
        for level in missing_levels:
            print(f"Running {interest_col} at {int(level*100)}% MNAR...")
            mnar_datasets, clean_df = create_mnar(df.copy(), interest_col, aux_cols, level)
            stats = evaluate_imputation(clean_df, mnar_datasets, interest_col, aux_cols)

            maes = extract_stat('MAE', {i:s for i, s in enumerate(stats)})
            mapes = extract_stat('MAPE', {i:s for i, s in enumerate(stats)})

            results.append({
                'column': interest_col,
                'missing_pct': level,
                'MAE_mean': np.mean(maes),
                'MAE_std': np.std(maes),
                'MAPE_mean': np.mean(mapes),
                'MAPE_std': np.std(mapes)
            })

    

    summary_df = pd.DataFrame(results)
    return summary_df


