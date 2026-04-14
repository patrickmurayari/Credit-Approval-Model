# Aprobación de Créditos - Modelo Predictivo

## Descripción

Proyecto de Machine Learning para predecir la aprobación de préstamos bancarios basado en características del solicitante. Utiliza un modelo **HistGradientBoostingClassifier** optimizado mediante GridSearchCV y SMOTE para el balanceo de clases.

## Resultados del Modelo

| Modelo | F1-score (Test) | Precision | Recall |
|--------|:--------------:|:---------:|:------:|
| **Gradient Boosting** | **0.82** | **0.87** | **0.87** |
| Random Forest | 0.80 | 0.84 | 0.88 |
| XGBoost | 0.76 | 0.84 | 0.81 |

### Variables más importantes (Permutation Importance)

1. **Credit_History** — Historial crediticio del solicitante (variable más predictiva)
2. **ApplicantIncome** — Ingreso del solicitante
3. **CoapplicantIncome** — Ingreso del co-solicitante
4. **LoanAmount** — Monto del préstamo solicitado

El modelo reducido con solo estas 4 variables obtuvo un **F1-score de 0.8471** y un **AP de 0.87**, demostrando que la reducción de dimensionalidad no afecta significativamente el rendimiento.

## Estructura del Proyecto

```
Credit_Approval/
├── main.py                  # Pipeline secuencial
├── requirements.txt         # Dependencias
├── README.md
├── data/
│   ├── raw/                 # Datos originales
│   └── processed/           # Dataset limpio
├── models/                  # Modelos entrenados (.pkl)
├── notebooks/               # Jupyter notebooks
├── reports/
│   └── figures/             # Gráficos del EDA y métricas
└── src/
    ├── __init__.py
    ├── data_cleaning.py     # Carga + limpieza + guardado
    ├── eda.py               # Análisis exploratorio + gráficos
    └── train.py             # Entrenamiento + evaluación + guardado
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/Credit_Approval.git
cd Credit_Approval

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Ejecutar pipeline completo

```bash
python main.py
```

### Ejecutar módulos individualmente

```bash
# Solo limpieza de datos
python src/data_cleaning.py

# Solo EDA
python src/eda.py

# Solo entrenamiento
python src/train.py
```

## Pipeline

1. **Limpieza de datos** (`src/data_cleaning.py`): Carga desde `data/raw/`, imputación de nulos, codificación de variables, transformación logarítmica de ingresos, guardado en `data/processed/`.

2. **EDA** (`src/eda.py`): Genera histogramas, boxplots, heatmap de correlación y gráficos de proporción. Todos los gráficos se guardan en `reports/figures/`.

3. **Entrenamiento** (`src/train.py`): Split train/test, SMOTE para balanceo, GridSearchCV para Random Forest, Gradient Boosting y XGBoost. Permutation Importance, curvas Precision-Recall y matrices de confusión. El modelo final se guarda en `models/`.

## Decisiones de Preprocesamiento

- **Credit_History**: Los valores faltantes se codifican como -1 (nueva categoría) para evitar sesgo por imputación con moda en una variable desbalanceada.
- **LoanAmount**: Imputación por mediana (distribución con asimetría 2.68 y curtosis 10.40).
- **Ingresos**: Transformación `log1p` para reducir el sesgo positivo y la influencia de outliers.
- **Property_Area**: One-Hot Encoding con `drop_first=True` para evitar multicolinealidad.

## Tecnologías

- Python 3.10+
- scikit-learn, XGBoost, imbalanced-learn
- pandas, numpy, statsmodels
- matplotlib, seaborn, shap
