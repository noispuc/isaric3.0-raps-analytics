"""
logistic_regression.py

Example script for running logistic regression and evaluating model assumptions.

This script demonstrates:
- How to load and prepare data
- How to run a logistic regression using execute_glm_regression
- How to visualize results with a Forest Plot
- How to validate model assumptions (Binary Dependent Variable, No Multicollinearity, Linearity of the Logit, Independence of Observations)
- How to assess predictive performance using Accuracy, Log Loss, Confusion Matrix, F1 Score, ROC Curve, and cross-validation
"""



from .core import execute_glm_regression



# Uploading data set
def logistic_regression(df, outcome,predictors):

    """
    Fit a logistic regression (Binomial GLM) and return only the formatted results table.

    Args:
        df (pd.DataFrame): DataFrame containing all variables.
        outcome (str): Name of the binary outcome column (0/1 or a two-level category).
        predictors (Sequence[str] | str): Predictor column names (a list/tuple or a single string).

    Returns:
        pd.DataFrame: A table with the following columns:
            - "Study"
            - "OddsRatio"
            - "LowerCI"
            - "UpperCI"
            - "p-value"

    Raises:
        ValueError: If `predictors` is empty after normalization.
        KeyError: If `outcome` or any predictor is missing from `df`.

    Example:
        >>> res = logistic_regression(df, "label", ["age", "sex"])
        >>> res.head()
    """

    logistic_response = execute_glm_regression(
        elr_dataframe_df=df,
        elr_outcome_str=outcome,
        elr_predictors_list=predictors,
        model_type='logistic',
        print_results=False,
        labels=False,
        reg_type="Multi"
    )

    return logistic_response



  

