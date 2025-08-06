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

import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objs as go
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from core import execute_glm_regression
from utils import fig_forest_plot

# Load dataset
# Load dataset from local file
df = pd.read_csv("data/df_map.csv")

# Define outcome and predictors
outcome_variable = "vital_rr"
predictor_variables = ["demog_age", "demog_sex", "comor_hypertensi",
                       "comor_diabetes_yn", "vital_highesttem_c", "labs_creatinine_mgdl"]

# Run regression
summary_df = execute_glm_regression(
    elr_dataframe_df=df,
    elr_outcome_str=outcome_variable,
    elr_predictors_list=predictor_variables,
    model_type="linear",
    print_results=True,
    labels=False,
    reg_type="Multi"
)

# Forest Plot
graph = fig_forest_plot(
    df=summary_df,
    labels=summary_df.columns.tolist(),
    only_display=True
)
graph.show()

# Prepare data for model validation
X = df[predictor_variables]
X = pd.get_dummies(X, drop_first=True)
X = sm.add_constant(X)
y = df[outcome_variable].astype(float)
X = X.astype(float)

data = pd.concat([y, X], axis=1).dropna()
y_clean = data[y.name]
X_clean = data.drop(y.name, axis=1)

# Fit model
model = sm.OLS(y_clean, X_clean).fit()

# Linearity: Residuals vs Fitted
fig = go.Figure()
fig.add_trace(go.Scatter(x=model.fittedvalues, y=model.resid, mode='markers', marker=dict(color='blue', size=8)))
fig.add_hline(y=0, line_dash='dash', line_color='red')
fig.update_layout(title='Residuals vs Adjusted Values', xaxis_title='Adjusted Values',
                  yaxis_title='Residuals',
                  yaxis_range=[min(model.resid)*1.1, max(model.resid)*1.1])
fig.show()

# Independence: Durbin-Watson
dw = durbin_watson(model.resid)
print(f'Durbin-Watson Statistic: {dw:.3f}')
if dw < 1.5:
    print("→ Indicates positive autocorrelation of residuals.")
elif dw > 2.5:
    print("→ Indicates negative autocorrelation of residuals.")
else:
    print("→ Residuals are likely independent.")

# Normality: Q-Q Plot and Shapiro-Wilk
qq = stats.probplot(model.resid, dist="norm")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='blue', size=8)))
fig2.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', line=dict(color='red', dash='dash')))
fig2.update_layout(title='Normality of Errors: Q-Q Plot',
                   xaxis_title='Theoretical Quantiles',
                   yaxis_title='Sample Quantiles')
fig2.show()

shapiro_test = stats.shapiro(model.resid)
print(f"Shapiro–Wilk test statistic: {shapiro_test.statistic:.4f}")
print(f"Shapiro–Wilk p-value: {shapiro_test.pvalue:.4f}")
if shapiro_test.pvalue > 0.05:
    print("Residuals appear to be normally distributed (fail to reject H0).")
else:
    print("Residuals do not appear to be normally distributed (reject H0).")

# Multicollinearity: VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data = vif_data[vif_data["Variable"] != "const"]
print(vif_data)

# Influential Outliers: Cook’s Distance
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
threshold = 4 / len(cooks_d)
influential_points = [i for i, val in enumerate(cooks_d) if val > threshold]
print(f"Above limit points ({threshold:.3f}): {influential_points}")

# Predictive Quality Metrics
mse = mean_squared_error(y_clean, model.fittedvalues)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_clean, model.fittedvalues)
r2 = r2_score(y_clean, model.fittedvalues)
n = X_clean.shape[0]
p = X_clean.shape[1]
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R² (Coefficient of Determination):", r2)
print("Adjusted R²:", adjusted_r2)

# Cross-Validation
model_cv = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
cv_mse_scores = -cross_val_score(model_cv, X_clean, y_clean, cv=kf, scoring=mse_scorer)

print("Cross-Validation Mean Squared Errors (MSE):", cv_mse_scores)
print("Mean CV MSE:", np.mean(cv_mse_scores))
print("Standard Deviation of CV MSE:", np.std(cv_mse_scores))
