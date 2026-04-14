import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import run_cleaning
from src.eda import run_eda
from src.train import run_training


def main():
    start = time.time()
    print("=" * 60)
    print("  PIPELINE DE APROBACIÓN DE CRÉDITOS")
    print("=" * 60)

    # Etapa 1: Limpieza de datos
    print("\n[1/3] Limpieza de datos...")
    run_cleaning()

    # Etapa 2: Análisis Exploratorio
    print("\n[2/3] Generando EDA...")
    run_eda()

    # Etapa 3: Entrenamiento del modelo
    print("\n[3/3] Entrenamiento y evaluación del modelo...")
    run_training()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETADO en {elapsed:.1f} segundos")
    print("=" * 60)


if __name__ == "__main__":
    main()
