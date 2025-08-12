import pandas as pd

def preprocess_cox_data(df, duration_col, event_col, predictors):
    """
    Pré-processa os dados para regressão de Cox:
    - Converte variáveis categóricas para dtype 'category'
    - Aplica one-hot encoding com drop_first
    - Garante que tempo e evento sejam numéricos
    - Remove linhas com dados faltantes

    Args:
        df (pd.DataFrame): Dataset bruto.
        duration_col (str): Coluna de tempo.
        event_col (str): Coluna de evento.
        predictors (list): Lista de variáveis preditoras.

    Returns:
        tuple: (DataFrame transformado, lista atualizada de preditores)
    """
    df = df.copy()

    # Identificar variáveis categóricas dentro dos preditores
    cat_vars = df.select_dtypes(include=["object", "category"]).columns.intersection(predictors)
    for col in cat_vars:
        df[col] = df[col].astype("category")

    # One-hot encoding
    df = pd.get_dummies(df, columns=cat_vars, drop_first=True)

    # Converter tempo e evento para numérico
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    # Atualizar lista de preditores (incluindo dummies geradas)
    updated_predictors = [c for c in df.columns if c in predictors or any(c.startswith(p + "_") for p in cat_vars)]

    # Remover linhas com NA nos campos relevantes
    df = df.dropna(subset=[duration_col, event_col] + updated_predictors)

    return df, updated_predictors


def summarize_model_output(summary_df, labels=None):
    """
    Aplica mapeamento de nomes legíveis à saída do modelo Cox.

    Args:
        summary_df (pd.DataFrame): Saída tabular do modelo.
        labels (dict, optional): Dicionário com mapeamento de nomes de variáveis.

    Returns:
        pd.DataFrame: Tabela com variáveis possivelmente renomeadas.
    """
    summary_df = summary_df.copy()
    if labels:
        summary_df["Variable"] = summary_df["Variable"].map(labels).fillna(summary_df["Variable"])
    return summary_df
