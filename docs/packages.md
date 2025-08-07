# 📦 Utilizando os Pacotes do Projeto

Este projeto utiliza **namespace packages**, permitindo uma estrutura modular e escalável sem arquivos `__init__.py`.

## 🧩 Estrutura
src/ 
└── raps/ 
        ├── preprocessing/ 
        ├── inference/ 
        └── predictive/
Cada subpacote contém módulos especializados (`core.py`, `utils.py`, etc.).

## 🚀 Instalação local

Para instalar o projeto e habilitar os imports como `raps.preprocessing`, execute:

```bash
pip install -e .
```
Certifique-se de estar na raiz do projeto e que os arquivos setup.cfg e rapsanalytics.toml estejam presentes.

## 🛠️ Exemplo de uso
```python
from raps.preprocessing import core
from raps.inference import survival_cox
from raps.predictive import linear_log_reg

df_clean = core.clean_data(df)
model = survival_cox.fit_model(df_clean)
preds = linear_log_reg.predict(model, df_clean)
```

##📚 Vantagens
Modularidade total

- Sem necessidade de __init__.py
- Fácil expansão com novos subpacotes
- Compatível com ferramentas modernas de empacotamento