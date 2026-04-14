import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _save_fig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado: {path}")


def load_processed(filename: str = "credit_processed.csv") -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / filename
    df = pd.read_csv(path)
    print(f"Dataset procesado cargado: {df.shape}")
    return df


def plot_credit_history_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["Credit_History"], bins=5, edgecolor="black")
    ax.set_xlabel("Credit_History")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de Credit_History")
    _save_fig(fig, "01_credit_history_histogram.png")


def plot_credit_history_vs_target(df: pd.DataFrame) -> None:
    sns.set(style="whitegrid", palette="pastel")
    var = "Credit_History"
    prop_df = df.groupby([var, "Loan_Status"], observed=True).size().reset_index(name="count")
    total_per_category = prop_df.groupby(var)["count"].transform("sum")
    prop_df["proportion"] = prop_df["count"] / total_per_category

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=prop_df, x=var, y="proportion", hue="Loan_Status", ax=ax)
    ax.set_title("Proporción de Loan_Status según Credit_History")
    ax.set_xlabel("Credit_History")
    ax.set_ylabel("Proporción")
    ax.set_ylim(0, 1)
    ax.legend(title="Loan_Status")
    _save_fig(fig, "02_credit_history_vs_target.png")


def plot_loan_amount_histograms(df: pd.DataFrame) -> None:
    columnas = ["LoanAmount", "Loan_Amount_Term"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, col in enumerate(columnas):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="black")
        axes[i].set_title(f"Histograma de {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frecuencia")
    plt.tight_layout()
    _save_fig(fig, "03_loan_amount_term_histograms.png")


def plot_loan_term_bar(df: pd.DataFrame) -> None:
    counts = df["Loan_Amount_Term"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index.astype(int).astype(str), counts.values, color="skyblue")
    ax.set_xlabel("Loan Amount Term (meses)")
    ax.set_ylabel("Cantidad de préstamos")
    ax.set_title("Distribución de Loan Amount Term")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    _save_fig(fig, "04_loan_term_bar_chart.png")


def plot_loan_amount_boxplot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df["LoanAmount"], ax=ax)
    ax.set_title("Boxplot de LoanAmount")
    ax.set_xlabel("Monto del Préstamo")
    _save_fig(fig, "05_loan_amount_boxplot.png")


def plot_income_histograms(df: pd.DataFrame) -> None:
    columnas = ["ApplicantIncome", "CoapplicantIncome"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, col in enumerate(columnas):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="black")
        axes[i].set_title(f"Histograma de {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frecuencia")
    plt.tight_layout()
    _save_fig(fig, "06_income_histograms.png")


def plot_income_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="ApplicantIncome", ax=axes[0])
    axes[0].set_title("Boxplot - ApplicantIncome")
    sns.boxplot(data=df, x="CoapplicantIncome", ax=axes[1])
    axes[1].set_title("Boxplot - CoapplicantIncome")
    plt.tight_layout()
    _save_fig(fig, "07_income_boxplots.png")


def plot_log_income_histograms(df: pd.DataFrame) -> None:
    columnas = ["Log_ApplicantIncome", "Log_CoapplicantIncome"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, col in enumerate(columnas):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="black")
        axes[i].set_title(f"Histograma de {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frecuencia")
    plt.tight_layout()
    _save_fig(fig, "08_log_income_histograms.png")


def plot_log_income_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x="Log_ApplicantIncome", ax=axes[0])
    axes[0].set_title("Boxplot - Log_ApplicantIncome")
    sns.boxplot(data=df, x="Log_CoapplicantIncome", ax=axes[1])
    axes[1].set_title("Boxplot - Log_CoapplicantIncome")
    plt.tight_layout()
    _save_fig(fig, "09_log_income_boxplots.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlación entre variables numéricas")
    _save_fig(fig, "10_correlation_heatmap.png")


def plot_self_employed_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="Self_Employed", hue="Self_Employed", ax=ax)
    ax.set_title("Distribución de Self_Employed")
    _save_fig(fig, "11_self_employed_distribution.png")


def run_eda() -> None:
    df = load_processed()
    print("Generando gráficos del EDA...")
    plot_credit_history_histogram(df)
    plot_credit_history_vs_target(df)
    plot_loan_amount_histograms(df)
    plot_loan_term_bar(df)
    plot_loan_amount_boxplot(df)
    plot_income_histograms(df)
    plot_income_boxplots(df)
    plot_log_income_histograms(df)
    plot_log_income_boxplots(df)
    plot_correlation_heatmap(df)
    plot_self_employed_hist(df)
    print("EDA completado. Todos los gráficos guardados en reports/figures/")


if __name__ == "__main__":
    run_eda()
