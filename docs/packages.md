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


# 📦 Installation & Usage Guide for `rapsanalytics`

This guide explains how to install and use the `rapsanalytics` package locally in your Python project.

---

## 🚀 Step 1: Copy the Package File

1. Locate the file:  
   `rapsanalytics-0.1.0-py3-none-any.whl`

2. Copy this file into the **root folder of your Python project** (where your `.py` scripts live).  
   This avoids path errors during installation.

---

## 🛠️ Step 2: Install the Package with `pip`

Open your terminal (e.g., VSCode integrated terminal) and run:

```bash
pip install ./rapsanalytics-0.1.0-py3-none-any.whl
✅ Make sure you're in the same directory as the .whl file when running this command.

If you're using a virtual environment, activate it first.

🧪 Step 3: Import and Use the Package
Once installed, you can import the package in your Python scripts like this:

python
import rapsanalytics as raps

# Example usage:
raps.preprocessing.clean_data()
raps.inference.run_model()
raps.prediction.forecast()
Note: If you're using a namespace package (no __init__.py), you may need to import submodules directly:

python
from rapsanalytics import preprocessing
preprocessing.clean_data()