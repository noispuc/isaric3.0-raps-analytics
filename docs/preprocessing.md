
---

### 3. `docs/preprocessing.md`

```markdown
# Módulo de Pré-processamento

Este módulo contém ferramentas para limpeza e imputação de dados, incluindo o método MICE.

## Arquivos

- `core.py`: Funções principais de pré-processamento.
- `mice.py`: Imputação multivariada com chained equations.
- `utils.py`: Funções auxiliares para manipulação de dados.

## Exemplo de uso

```python
from src.preprocessing.mice import run_mice_imputation

df_imputed = run_mice_imputation(df, variables=["idade", "creatinina"])
print(df_imputed.head())
