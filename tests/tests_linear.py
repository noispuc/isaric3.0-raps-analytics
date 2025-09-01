# tests/tests_linear.py
from pathlib import Path
import sys
import pandas as pd
# 1) aponta para a pasta raiz do repo e para ./src
ROOT = Path(__file__).resolve().parents[1]   # sobe de tests/ para a raiz
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))


# 2) importe do pacote (confirme 'predictive' vs 'predictives')
from raps.predictive.linear_log_reg import linear_regression
from raps.predictive.logistic_regression import logistic_regression
from raps.inference.survival_cox import survival_cox

# 3) smoke test
import numpy as np, pandas as pd
df_map = pd.read_csv('data/df_map.csv')
linear = linear_regression(df_map,"vital_rr",["demog_age", "demog_sex", "comor_hypertensi", "comor_diabetes_yn", "vital_highesttem_c", "labs_creatinine_mgdl"])

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

logistic = logistic_regression(df_map,outcome,predictors)

df_model = pd.read_csv('data/df_model.csv')

duration_col = 'HospitalLengthStay_trunc'  # Time variable
event_col = 'HospitalDischargeCode_trunc_bin'  # Binary outcome variable
cox_predictors = [
    'period', 'Idade_Agrupada2', 'ChronicHealthStatusName', 'obesity',
    'IsImmunossupression', 'IsSteroidsUse', 'IsSevereCopd', 'IsChfNyha',
    'cancer', 'ResourceIsRenalReplacementTherapy', 'ResourceIsVasopressors',
    'Vent_Resource'
]

cox = survival_cox(df_model,duration_col,event_col,cox_predictors)


#print("Linear Regression:")
#print(linear)
#print("Logistic Regression:")
#print(logistic)
print("Cox Model:")
print(cox)