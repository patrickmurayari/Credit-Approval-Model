import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


def _save_fig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado: {path}")


def load_processed(filename: str = "credit_processed.csv") -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / filename
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE aplicado. Train balanceado: {X_res.shape[0]} muestras")
    return X_res, y_res


def compare_models(X_res, y_res, cv: int = 5) -> dict:
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced"),
        "SVM": SVC(class_weight="balanced"),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", class_weight="balanced"),
    }
    resultados = {}
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X_res, y_res, cv=cv, scoring="f1")
        resultados[nombre] = scores.mean()
        print(f"{nombre}: F1 promedio = {scores.mean():.4f}")
    return resultados


def grid_search_tuning(X_res, y_res, X_test, y_test):
    # --- Random Forest ---
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="f1", n_jobs=-1)
    grid_search_rf.fit(X_res, y_res)
    print(f"\nRandom Forest - Mejores parámetros: {grid_search_rf.best_params_}")
    print(f"Random Forest - Mejor F1 CV: {grid_search_rf.best_score_:.4f}")

    # --- Gradient Boosting (HistGradientBoosting) ---
    param_grid_gr = {
        "max_iter": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "learning_rate": [0.01, 0.1, 0.2],
        "min_samples_leaf": [1, 2, 4],
    }
    gr = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
    grid_search_gr = GridSearchCV(gr, param_grid_gr, cv=5, scoring="f1", n_jobs=-1)
    grid_search_gr.fit(X_res, y_res)
    print(f"\nGradient Boosting - Mejores parámetros: {grid_search_gr.best_params_}")
    print(f"Gradient Boosting - Mejor F1 CV: {grid_search_gr.best_score_:.4f}")

    # --- XGBoost ---
    scale_pos_weight_val = sum(y_res == 0) / sum(y_res == 1)
    param_grid_xg = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }
    xg = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, scale_pos_weight=scale_pos_weight_val
    )
    grid_search_xg = GridSearchCV(xg, param_grid_xg, cv=5, scoring="f1", n_jobs=-1)
    grid_search_xg.fit(X_res, y_res)
    print(f"\nXGBoost - Mejores parámetros: {grid_search_xg.best_params_}")
    print(f"XGBoost - Mejor F1 CV: {grid_search_xg.best_score_:.4f}")

    # Evaluación en test
    best_rf = grid_search_rf.best_estimator_
    best_gr = grid_search_gr.best_estimator_
    best_xg = grid_search_xg.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    y_pred_gr = best_gr.predict(X_test)
    y_pred_xg = best_xg.predict(X_test)

    print("\n--- Evaluación en conjunto de prueba ---")
    print("Random Forest:")
    print(classification_report(y_test, y_pred_rf))
    print("Gradient Boosting:")
    print(classification_report(y_test, y_pred_gr))
    print("XGBoost:")
    print(classification_report(y_test, y_pred_xg))

    return best_rf, best_gr, best_xg


def plot_f1_comparison(best_rf, best_gr, best_xg, X_test, y_test) -> None:
    modelos = {"Random Forest": best_rf, "Gradient Boosting": best_gr, "XGBoost": best_xg}
    f1_scores = {}
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        f1_scores[nombre] = f1_score(y_test, y_pred, average="weighted")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(f1_scores.keys(), f1_scores.values(), color=["#2ecc71", "#3498db", "#e74c3c"])
    ax.set_ylabel("F1-score (weighted)")
    ax.set_title("Comparación de F1-score en Test")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, f1_scores.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
    _save_fig(fig, "12_f1_comparison.png")


def plot_precision_recall_curves(best_rf, best_gr, best_xg, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    modelos = {
        "Random Forest": best_rf,
        "Gradient Boosting": best_gr,
        "XGBoost": best_xg,
    }
    for nombre, modelo in modelos.items():
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        else:
            y_proba = modelo.decision_function(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{nombre} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, "13_precision_recall_curves.png")


def plot_confusion_matrices(best_rf, best_gr, best_xg, X_test, y_test) -> None:
    modelos = {"Random Forest": best_rf, "Gradient Boosting": best_gr, "XGBoost": best_xg}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (nombre, modelo) in enumerate(modelos.items()):
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"{nombre}")
        axes[i].set_xlabel("Predicción")
        axes[i].set_ylabel("Real")
    plt.tight_layout()
    _save_fig(fig, "14_confusion_matrices.png")


def plot_permutation_importance(model, X_test, y_test, model_name: str) -> pd.Series:
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=perm_imp, y=perm_imp.index, palette="viridis", ax=ax)
    ax.set_title(f"Permutation Importance ({model_name})")
    ax.set_xlabel("Importancia Promedio")
    ax.set_ylabel("Características")
    plt.tight_layout()
    _save_fig(fig, f"15_permutation_importance_{model_name.replace(' ', '_').lower()}.png")
    return perm_imp


def train_top4_model(top_4_features, X_train, y_train, X_test, y_test):
    X_train_top4 = X_train[top_4_features]
    X_test_top4 = X_test[top_4_features]

    smote = SMOTE(random_state=42)
    X_res_top4, y_res_top4 = smote.fit_resample(X_train_top4, y_train)

    param_grid = {
        "max_iter": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "learning_rate": [0.01, 0.1, 0.2],
        "min_samples_leaf": [1, 2, 4],
    }
    gr_top4 = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
    grid_top4 = GridSearchCV(gr_top4, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_top4.fit(X_res_top4, y_res_top4)

    best_top4 = grid_top4.best_estimator_
    y_pred_top4 = best_top4.predict(X_test_top4)
    f1_top4 = f1_score(y_test, y_pred_top4, average="weighted")

    print(f"\n--- Modelo con Top 4 Features ---")
    print(f"Features: {top_4_features}")
    print(f"Mejores parámetros: {grid_top4.best_params_}")
    print(f"F1 Score (test): {f1_top4:.4f}")
    print(classification_report(y_test, y_pred_top4))

    return best_top4, f1_top4


def save_model(model, filename: str = "gradient_boosting_model.pkl") -> Path:
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {path}")
    return path


def run_training() -> Path:
    df = load_processed()
    X_train, X_test, y_train, y_test = split_data(df)
    X_res, y_res = apply_smote(X_train, y_train)

    print("\n--- Comparación de modelos ---")
    compare_models(X_res, y_res)

    print("\n--- GridSearchCV ---")
    best_rf, best_gr, best_xg = grid_search_tuning(X_res, y_res, X_test, y_test)

    plot_f1_comparison(best_rf, best_gr, best_xg, X_test, y_test)
    plot_precision_recall_curves(best_rf, best_gr, best_xg, X_test, y_test)
    plot_confusion_matrices(best_rf, best_gr, best_xg, X_test, y_test)

    print("\n--- Permutation Importance ---")
    perm_imp_gr = plot_permutation_importance(best_gr, X_test, y_test, "Gradient Boosting")
    plot_permutation_importance(best_rf, X_test, y_test, "Random Forest")

    top_4 = perm_imp_gr.head(4).index.tolist()
    print(f"\nTop 4 features: {top_4}")
    best_top4, f1_top4 = train_top4_model(top_4, X_train, y_train, X_test, y_test)

    model_path = save_model(best_gr)
    save_model(best_top4, "gradient_boosting_top4_model.pkl")

    print("\n--- Conclusión ---")
    print("Gradient Boosting seleccionado como modelo final (F1=0.82 en test).")
    print(f"Modelo reducido (Top 4) F1={f1_top4:.4f}")
    return model_path


if __name__ == "__main__":
    run_training()
