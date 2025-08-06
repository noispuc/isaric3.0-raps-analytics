
---

### 2. `docs/predictive.md`

```markdown
# Módulo Preditivo

Este módulo oferece ferramentas para regressão linear e logística, além de validação de modelos e visualização com Forest Plot.

## Arquivos

- `core.py`: Função `execute_glm_regression` para regressão GLM.
- `linear_log_reg.py`: Exemplo completo de regressão linear com validação.
- `utils.py`: Geração de Forest Plot interativo com Plotly.

## Exemplo de uso

```python
from src.predictive.core import execute_glm_regression

summary_df = execute_glm_regression(
    elr_dataframe_df=df,
    elr_outcome_str="vital_rr",
    elr_predictors_list=["idade", "sexo"],
    model_type="linear"
)
print(summary_df)
