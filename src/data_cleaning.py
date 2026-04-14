import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data(filename: str = "Grupo 1 - Aprobación de Créditos.csv") -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "raw" / filename
    df = pd.read_csv(path)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminar Loan_ID (identificador, no predictor)
    df = df.drop(columns=["Loan_ID"])

    # Convertir columnas object a categóricas
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.Categorical(df[col])

    # Imputación de variables categóricas con la moda
    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Credit_History: crear categoría -1 para datos faltantes
    df["Credit_History"] = df["Credit_History"].fillna(-1).astype(int)

    # Loan_Amount_Term: imputar con la moda
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

    # LoanAmount: imputar con la mediana (distribución sesgada)
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    # Dependents: convertir '3+' a 3 y pasar a entero
    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    # LabelEncoder para variables binarias o casi-binarias
    label_cols = ["Gender", "Married", "Education", "Self_Employed", "Loan_Status"]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col])

    # One-Hot Encoding para Property_Area (más de 2 categorías)
    df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

    # Convertir columnas booleanas de get_dummies a enteros
    bool_cols = df.select_dtypes(include="bool").columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Transformación logarítmica de ingresos (reducción de sesgo)
    df["Log_ApplicantIncome"] = np.log1p(df["ApplicantIncome"])
    df["Log_CoapplicantIncome"] = np.log1p(df["CoapplicantIncome"])

    print(f"Datos limpios: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Valores nulos restantes:\n{df.isnull().sum()}")
    return df


def save_processed(df: pd.DataFrame, filename: str = "credit_processed.csv") -> Path:
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Dataset procesado guardado en: {out_path}")
    return out_path


def run_cleaning() -> Path:
    df = load_data()
    df_clean = clean_data(df)
    return save_processed(df_clean)


if __name__ == "__main__":
    run_cleaning()
