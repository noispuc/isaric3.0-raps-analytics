import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, roc_auc_score

def fit_cox_model(df, duration_col, event_col, predictors):
    """
    Ajusta o modelo de Cox Proporcional usando lifelines.

    Args:
        df (pd.DataFrame): Dataset contendo todas as variáveis.
        duration_col (str): Coluna com o tempo até o evento.
        event_col (str): Coluna binária com o evento (1=ocorreu, 0=censurado).
        predictors (list): Lista de colunas preditoras.

    Returns:
        CoxPHFitter: Modelo Cox treinado.
    """
    df_model = df[[duration_col, event_col] + predictors].copy()
    cph = CoxPHFitter()
    cph.fit(df_model, duration_col=duration_col, event_col=event_col)
    return cph

def extract_model_summary(cph_model):
    """
    Extrai estatísticas do modelo Cox: HR, IC 95%, p-valor.

    Args:
        cph_model (CoxPHFitter): Modelo ajustado.

    Returns:
        pd.DataFrame: Tabela com variáveis, HR, IC e p-valores.
    """
    summary = cph_model.summary.copy()
    summary['HR'] = np.exp(summary['coef'])
    summary['CI_lower'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    summary['CI_upper'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
    summary['p_adj'] = summary['p'].apply(lambda p: "<0.001" if p < 0.001 else round(p, 3))
    
    df_result = summary[['HR', 'CI_lower', 'CI_upper', 'p_adj']].reset_index()
    df_result.rename(columns={'index': 'Variable', 'p_adj': 'p-value'}, inplace=True)
    return df_result

def calculate_c_index(cph_model, df, duration_col, event_col):
    """
    Calcula o índice de concordância (C-index).

    Args:
        cph_model (CoxPHFitter): Modelo ajustado.
        df (pd.DataFrame): Dados usados no ajuste.
        duration_col (str): Coluna tempo.
        event_col (str): Coluna evento.

    Returns:
        float: Valor de C-index.
    """
    scores = -cph_model.predict_partial_hazard(df)
    return concordance_index(df[duration_col], scores, df[event_col])

def plot_roc_auc(cph_model, df, duration_col, event_col, time_point):
    """
    Plota curva ROC e calcula AUC para um tempo específico.

    Args:
        cph_model (CoxPHFitter): Modelo ajustado.
        df (pd.DataFrame): Dataset original.
        duration_col (str): Tempo até evento.
        event_col (str): Indicador de evento.
        time_point (float): Tempo de corte para análise.

    Returns:
        None
    """
    risk = cph_model.predict_partial_hazard(df)
    cases = (df[duration_col] <= time_point) & (df[event_col] == 1)
    controls = df[duration_col] > time_point
    mask = cases | controls

    y_true = cases[mask]
    y_score = risk[mask]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (0.5)')
    plt.title(f'Curva ROC - t={time_point}')
    plt.xlabel('Falso Positivo'); plt.ylabel('Verdadeiro Positivo')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_calibration(cph_model, df, duration_col, event_col, t):
    """
    Plota gráfico de calibração comparando previsão e observação por decil.

    Args:
        cph_model (CoxPHFitter): Modelo ajustado.
        df (pd.DataFrame): Dataset.
        duration_col (str): Tempo.
        event_col (str): Evento.
        t (float): Tempo de avaliação.

    Returns:
        None
    """
    pred_surv = cph_model.predict_survival_function(df, times=[t]).squeeze()
    df_calib = pd.DataFrame({
        'predicted_survival': pred_surv,
        'duration': df[duration_col],
        'event': df[event_col]
    })

    df_calib['decile'] = pd.qcut(df_calib['predicted_survival'], 10, labels=False, duplicates='drop')

    obs_probs, pred_probs = [], []
    for i in sorted(df_calib['decile'].unique()):
        subset = df_calib[df_calib['decile'] == i]
        pred_probs.append(subset['predicted_survival'].mean())

        kmf = KaplanMeierFitter()
        kmf.fit(subset['duration'], event_observed=subset['event'])
        obs_probs.append(kmf.predict(t))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Calibração perfeita')
    plt.plot(pred_probs, obs_probs, 'o-', label='Modelo')
    plt.title(f'Calibração - t={t}')
    plt.xlabel('Probabilidade prevista'); plt.ylabel('Sobrevivência observada')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def evaluate_brier_score(cph_model, df, duration_col, event_col, target_time, plot=True):
    """
    Avalia e plota o Brier Score para previsão de sobrevivência.

    Args:
        cph_model (CoxPHFitter): Modelo ajustado.
        df (pd.DataFrame): Dataset.
        duration_col (str): Tempo.
        event_col (str): Evento.
        target_time (float): Tempo alvo.
        plot (bool): Plota o gráfico (default = True).

    Returns:
        tuple: Vetores de tempo e Brier scores.
    """
    T = df[duration_col]
    E = df[event_col]
    min_time = T[E == 1].min()
    max_time = min(target_time, T.max())
    time_points = np.linspace(min_time, max_time, 100)

    kmf_censoring = KaplanMeierFitter().fit(T, 1 - E)
    G_T = kmf_censoring.predict(T)

    scores = []
    for t in time_points:
        pred_probs = cph_model.predict_survival_function(df, times=[t]).squeeze()
        G_t = kmf_censoring.predict(t)

        mask_event = (T <= t) & (E == 1)
        mask_survived = T > t

        term1 = np.sum(((pred_probs[mask_event] - 0) ** 2) / G_T[mask_event])
        term2 = np.sum(((pred_probs[mask_survived] - 1) ** 2) / G_t)

        score = (term1 + term2) / len(df)
        scores.append(score)

    if plot:
        idx = np.argmin(np.abs(time_points - target_time))
        score_t = scores[idx]

        plt.figure(figsize=(10, 6))
        plt.plot(time_points, scores)
        plt.plot(target_time, score_t, 'ro', label=f't={target_time}: {score_t:.4f}')
        plt.axhline(0.25, linestyle='--', color='gray', label='Modelo não informativo (0.25)')
        plt.title('Brier Score - Cox')
        plt.xlabel('Tempo'); plt.ylabel('Brier Score')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    return time_points, scores

def display_model_fit_metrics(cph_model, df):
    """
    Exibe métricas de ajuste do modelo Cox: AIC parcial e BIC parcial.

    Args:
        cph_model (CoxPHFitter): Modelo ajustado.
        df (pd.DataFrame): Dataset usado no ajuste.

    Returns:
        None
    """
    aic = cph_model.AIC_partial_
    n = len(df)
    k = len(cph_model.params_)
    loglik = cph_model.log_likelihood_
    bic = -2 * loglik + k * np.log(n)

    print(f"Partial AIC: {aic:.2f}")
    print(f"Partial BIC: {bic:.2f}")
