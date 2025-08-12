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
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .core import execute_glm_regression
from .utils import fig_forest_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score, recall_score, f1_score,  roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Uploading data set

df_map = pd.read_csv("data/df_map.csv")

# Creating binary variables for age, as age is between 0-110 in the normal dataset.
bins = list(range(0, 110, 10))
labels = [f"{i}_{i+10}" for i in bins[:-1]]

df_map['age_bin'] = pd.cut(df_map['demog_age'], bins=bins, labels=labels, right=False)

age_dummies = pd.get_dummies(df_map['age_bin'], prefix='age')
df_map = pd.concat([df_map, age_dummies], axis=1)

# Choosing one category for outcome, as this is not a multinomial regression.
df_map['outcome_binary'] = (df_map['outco_binary_outcome'] == 'Death').astype(int)

# Defining outcome and predictors variables
outcome = "outcome_binary"
predictors = ["demog_sex" , "age_bin"]

logistic_response = execute_glm_regression(
    elr_dataframe_df=df_map,
    elr_outcome_str=outcome,
    elr_predictors_list=predictors,
    model_type='logistic',
    print_results=True,
    labels=False,
    reg_type="Multi"
)






graph = fig_forest_plot(
    df = logistic_response,
    labels = logistic_response.columns.tolist(),
    only_display=True
)

graph.show()

print(df_map[outcome].value_counts())
print(df_map[outcome].unique())

from statsmodels.stats.outliers_influence import variance_inflation_factor


X = pd.get_dummies(df_map[["demog_sex", "age_bin"]], drop_first=True)


X = sm.add_constant(X)

X = X.astype(int)

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data = vif_data[vif_data["Variable"] != "const"]
print(vif_data)


n_events = df_map[outcome].sum()
n_predictors = len(predictors)
epv = n_events / n_predictors
print(f"EPV: {epv:.2f}")




y = df_map["outcome_binary"].astype(float)

model = sm.Logit(y, X).fit(maxiter=100)


influence = model.get_influence()


cooks_d = influence.cooks_distance[0]


leverage = influence.hat_matrix_diag


pearson_resid = model.resid_pearson
deviance_resid = model.resid_dev


dfbetas = influence.dfbetas


influential_points = np.where(cooks_d > 4 / model.nobs)[0]

print(f"Number of influential points (Cook's D > 4/n): {len(influential_points)}")
print("Top 5 Cook's distances:")
print(cooks_d[influential_points][:5])


from sklearn.metrics import accuracy_score

y_pred_class = (model.predict() >= 0.1017).astype(int) #threshold adjustment: 0.5 -> 0.1017
accuracy = accuracy_score(y, y_pred_class)
print("Accuracy:", accuracy)

from sklearn.metrics import log_loss

logloss = log_loss(y, model.predict())
print("Log Loss:", logloss)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_pred_class)
print("Confusion Matrix:\n", cm)

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y, y_pred_class, zero_division=0)
recall    = recall_score(y, y_pred_class, zero_division=0)
f1        = f1_score(y, y_pred_class, zero_division=0)

print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

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




# Re-create X and y for scikit-learn
X_cv = pd.get_dummies(df_map[["demog_sex", "age_bin"]], drop_first=True).astype(float)
y_cv = df_map["outcome_binary"].astype(int)

clf = LogisticRegression(max_iter=1000)
scores = cross_val_score(clf, X_cv, y_cv, cv=5, scoring="accuracy")

print("Cross-Validation Accuracies:", scores)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())
