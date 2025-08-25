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


import numpy as np
from .core import execute_glm_regression
from .utils import fig_forest_plot

# Load dataset
# Load dataset from local file

def linear_regression(df,outcome_variable,predictor_variables):
  
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



'''
    # Normality: Q-Q Plot and Shapiro-Wilk
    qq = stats.probplot(model.resid, dist="norm")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='blue', size=8)))
    fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', line=dict(color='red', dash='dash')))
    fig2.update_layout(title='Normality of Errors: Q-Q Plot',
                    xaxis_title='Theoretical Quantiles',
                    yaxis_title='Sample Quantiles')
   # fig2.show()

       # Linearity: Residuals vs Fitted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model.fittedvalues, y=model.resid, mode='markers', marker=dict(color='blue', size=8)))
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.update_layout(title='Residuals vs Adjusted Values', xaxis_title='Adjusted Values',
                    yaxis_title='Residuals',
                    yaxis_range=[min(model.resid)*1.1, max(model.resid)*1.1])
    #fig.show()

        # Forest Plot
    graph = fig_forest_plot(
        df=summary_df,
        labels=summary_df.columns.tolist(),
        only_display=True
    )


'''