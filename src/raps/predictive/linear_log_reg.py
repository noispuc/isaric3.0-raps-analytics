"""
linear_log_reg.py

Example script for running linear regression using GLM and evaluating model assumptions.

This script demonstrates:
- How to load and prepare data
- How to run a linear regression using execute_glm_regression
- How to visualize results with a Forest Plot
- How to validate model assumptions (linearity, homoscedasticity, normality, independence)
- How to assess predictive performance using MSE, RMSE, MAE, R², and cross-validation
"""



from .core import execute_glm_regression



def linear_regression(df,outcome_variable,predictor_variables):
    """
    Fit a linear regression (Gaussian GLM) and return only the formatted results table.

    Args:
        df (pd.DataFrame): DataFrame containing all variables.
        outcome_variable (str): Name of the continuous outcome column.
        predictor_variables (Sequence[str] | str): Predictor column names (a list/tuple or a single string).

    Returns:
        pd.DataFrame: A table with the following columns:
            - "Study"
            - "Coefficient"
            - "LowerCI"
            - "UpperCI"
            - "p-value"

    Raises:
        ValueError: If `predictor_variables` is empty after normalization.
        KeyError: If `outcome_variable` or any predictor is missing from `df`.

    Example:
         res = linear_regression(df, "y", ["x1", "x2"])
         res.head()
    """
    if isinstance(predictor_variables, list):
        npredictor = 0
        for predictor in predictor_variables:
            npredictor += 1
        if npredictor > 1:
            regType = "Multi"
        else:
            regType = "Uni"
    else:
        predictor_variables = [predictor_variables]
        regType = "Uni"

    # Run regression
    summary_df = execute_glm_regression(
        elr_dataframe_df=df,
        elr_outcome_str=outcome_variable,
        elr_predictors_list=predictor_variables,
        model_type="linear",
        print_results=False,
        labels=False,
        reg_type=regType
    )







    return summary_df


