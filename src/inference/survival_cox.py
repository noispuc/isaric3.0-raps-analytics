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

import warnings
import pandas as pd
from lifelines.exceptions import ApproximationWarning

from core import (
    fit_cox_model,
    extract_model_summary,
    calculate_c_index,
    plot_roc_auc,
    plot_calibration,
    evaluate_brier_score,
    display_model_fit_metrics
)

from utils import (
    preprocess_cox_data,
    summarize_model_output
)

def main():
    """
    Executa o pipeline completo de regressão de Cox.

    Etapas:
    -------
    1. Carrega os dados do arquivo 'df_model.csv'
    2. Pré-processa os dados (encoding e limpeza)
    3. Ajusta o modelo de Cox
    4. Gera tabela de Hazard Ratios com IC e p-valores
    5. Avalia o modelo: C-index, AUC, Brier Score, calibração
    6. Exibe métricas de ajuste: AIC e BIC

    Observação:
    -----------
    Os gráficos são exibidos automaticamente durante a execução.
    """
    # 1. Carregamento dos dados
    df = pd.read_csv("data/df_model.csv")

    # 2. Definições de variáveis
    duration_col = "HospitalLengthStay_trunc"
    event_col = "HospitalDischargeCode_trunc_bin"
    predictors = [
        "period", "Idade_Agrupada2", "ChronicHealthStatusName", "obesity",
        "IsImmunossupression", "IsSteroidsUse", "IsSevereCopd", "IsChfNyha",
        "cancer", "ResourceIsRenalReplacementTherapy", "ResourceIsVasopressors",
        "Vent_Resource"
    ]

    print("\n🔄 Pré-processando os dados...")
    df_processed, updated_predictors = preprocess_cox_data(df, duration_col, event_col, predictors)

    print("\n⚙️ Ajustando o modelo de Cox...")
    model = fit_cox_model(df_processed, duration_col, event_col, updated_predictors)

    print("\n📋 Resumo do modelo:")
    summary_df = extract_model_summary(model)
    print(summarize_model_output(summary_df))

    print("\n📊 Avaliação de desempenho:")
    print(f"C-Index: {calculate_c_index(model, df_processed, duration_col, event_col):.3f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ApproximationWarning)

        print("\n📈 Curva ROC:")
        plot_roc_auc(model, df_processed, duration_col, event_col, time_point=60)

        print("\n📐 Curva de calibração:")
        plot_calibration(model, df_processed, duration_col, event_col, t=60)

        print("\n📉 Brier Score:")
        evaluate_brier_score(model, df_processed, duration_col, event_col, target_time=60)

    print("\n📏 Métricas de ajuste:")
    display_model_fit_metrics(model, df_processed)

if __name__ == "__main__":
    main()
