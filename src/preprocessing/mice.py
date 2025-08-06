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
import seaborn as sns
import matplotlib.pyplot as plt
from .core import create_mnar, evaluate_imputation
from .utils import extract_stat

def main():
    """
    Executa a validação do MICE em dados reais com padrão MNAR simulado.

    Para cada coluna de interesse, este pipeline:
    - Gera 30 replicações com MNAR em níveis variados (1%, 5%, 10%, 20%, 30%)
    - Imputa os valores faltantes usando IterativeImputer (MICE)
    - Calcula métricas de erro (MAE, MAPE)
    - Plota boxplots das métricas por coluna e nível de ausência
    - Gera um resumo estatístico final

    Os resultados são impressos no console e exibidos como gráficos.

    Requisitos:
    -----------
    O arquivo 'dados_uti_ems.xlsx' deve estar disponível na raiz do projeto.
    """
    df = pd.read_excel('data/dados_uti_ems.xlsx')
    cols_to_test = ['Age', 'LengthHospitalStayPriorUnitAdmission', 'SofaScore', 'los', 'Saps3Points']
    missing_levels = [0.01, 0.05, 0.10, 0.20, 0.30]
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

            sns.boxplot(y=maes)
            plt.title(f"MAE Boxplot - {interest_col} - {int(level*100)}%")
            plt.ylabel("MAE")
            plt.show()

            if not pd.isna(np.mean(mapes)) and len(mapes) > 0:
                sns.boxplot(y=mapes)
                plt.title(f"MAPE Boxplot - {interest_col} - {int(level*100)}%")
                plt.ylabel("MAPE")
                plt.show()
            else:
                print(f"MAPE boxplot skipped for {interest_col} at {int(level*100)}% (empty or NaN values)")

    summary_df = pd.DataFrame(results)
    print(summary_df)

if __name__ == "__main__":
    main()
