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

import numpy as np

from .core import execute_glm_regression
from .utils import fig_forest_plot


# Uploading data set
def logistic_regression(df, outcome,predictors):



    logistic_response = execute_glm_regression(
        elr_dataframe_df=df,
        elr_outcome_str=outcome,
        elr_predictors_list=predictors,
        model_type='logistic',
        print_results=True,
        labels=False,
        reg_type="Multi"
    )

    return logistic_response



  


'''
    graph = fig_forest_plot(
        df = logistic_response,
        labels = logistic_response.columns.tolist(),
        only_display=True
    )

        from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt



    y_scores = model.predict()
    auc = roc_auc_score(y, y_scores)
    print(f"ROC AUC Score: {auc:.3f}")


    fpr, tpr, thresholds = roc_curve(y, y_scores)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
'''