"""
Cox Regression Orchestrator.

Este módulo organiza a execução da análise de sobrevivência com regressão de Cox
em dados clínicos, incluindo ajuste do modelo, avaliação de desempenho e diagnóstico
de suposições. Utiliza como base o arquivo 'df_model.csv'.

Uso recomendado:
---------------
Executar como script principal:
    python survival_cox.py

Ou importar e chamar a função `main()` diretamente.

Dependências:
-------------
- pandas
- numpy
- lifelines
- matplotlib
- módulo interno: core.py
- módulo interno: utils.py
"""




from .core import (
    fit_cox_model,
    extract_model_summary,
)

from .utils import (
    preprocess_cox_data,

)




def survival_cox(df,duration_col,event_col,predictors):
        """
    Run a minimal Cox proportional hazards pipeline (preprocess → fit → summarize).

    Args:
        df (pd.DataFrame): DataFrame containing all variables.
        duration_col (str): Column with time-to-event (or censoring) values.
        event_col (str): Binary event indicator column (1 = event occurred, 0 = censored).
        predictors (Sequence[str] | str): Predictor column names (a list/tuple or a single string).

    Returns:
        pd.DataFrame: Summary table with columns:
            - "covariate"
            - "HR"
            - "CI_lower"
            - "CI_upper"
            - "p-value"

    Raises:
        ValueError: If `predictors` is empty after normalization or if, after preprocessing,
                    no valid predictors remain.
        KeyError: If `duration_col`, `event_col`, or any requested predictor is missing in `df`.

    Example:
        >>> tbl = survival_cox(df, "time", "event", ["age", "sex", "spo2"])
        >>> tbl.head()
    """

        df_processed, updated_predictors = preprocess_cox_data(df, duration_col, event_col, predictors)

        model = fit_cox_model(df_processed, duration_col, event_col, updated_predictors)

        summary_df = extract_model_summary(model)

        return summary_df



       


