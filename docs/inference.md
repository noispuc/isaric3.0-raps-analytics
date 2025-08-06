# Módulo de Inferência

Este módulo contém ferramentas para análise estatística inferencial, incluindo regressão de Cox para dados de sobrevivência.

## Arquivos

- `core.py`: Funções principais para modelagem inferencial.
- `survival_cox.py`: Implementação da regressão de Cox.
- `utils.py`: Funções auxiliares para visualização e manipulação de dados.

## Exemplo de uso

```python
from src.inference.survival_cox import execute_cox_model

summary_df = execute_cox_model(df, outcome="tempo_morte", event="óbito", predictors=["idade", "sexo"])
print(summary_df)
