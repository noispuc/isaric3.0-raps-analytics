# tests/tests_linear.py
from pathlib import Path
import sys
import pandas as pd
# 1) aponta para a pasta raiz do repo e para ./src
ROOT = Path(__file__).resolve().parents[1]   # sobe de tests/ para a raiz
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))
print("Usando SRC =", SRC)  # opcional: só p/ conferir

# 2) importe do pacote (confirme 'predictive' vs 'predictives')
from raps.predictive.linear_log_reg import linear_regression

# 3) smoke test
import numpy as np, pandas as pd
df = pd.read_csv('data/df_map.csv')
print(linear_regression(df,"vital_rr",["demog_age", "demog_sex", "comor_hypertensi", "comor_diabetes_yn", "vital_highesttem_c", "labs_creatinine_mgdl"]))
