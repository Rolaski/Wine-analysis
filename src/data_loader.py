"""
Moduł odpowiedzialny za wczytywanie i podstawowe operacje na zbiorze danych Wine.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple


def load_wine_dataset(data_path: str = "../data/wine.data") -> pd.DataFrame:
    """
    Wczytuje zbiór danych Wine Dataset i dodaje odpowiednie nagłówki.
    Wywoływana na początku aplikacji.

    Args:
        data_path: Ścieżka do pliku z danymi

    Returns:
        DataFrame z wczytanymi danymi i nagłówkami
    """
    # Nazwy kolumn zgodnie z opisem z pliku wine.names
    column_names = [
        'Class',
        'Alcohol',
        'Malic acid',
        'Ash',
        'Alcalinity of ash',
        'Magnesium',
        'Total phenols',
        'Flavanoids',
        'Nonflavanoid phenols',
        'Proanthocyanins',
        'Color intensity',
        'Hue',
        'OD280/OD315 of diluted wines',
        'Proline'
    ]

    # Wczytanie danych z dodaniem nagłówków
    try:
        wine_df = pd.read_csv(data_path, header=None, names=column_names)
        print(f"Wczytano dane z {data_path} pomyślnie.")
        return wine_df
    except FileNotFoundError:
        print(f"Plik {data_path} nie istnieje.")
        return None
    except Exception as e:
        print(f"Wystąpił błąd podczas wczytywania danych: {e}")
        return None


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Zwraca podstawowe informacje o zbiorze danych.
    Wykorzystywany w sekcji -> informacje o zbiorze danych.

    Args:
        df: DataFrame z danymi

    Returns:
        Słownik z informacjami o zbiorze danych
    """
    if df is None:
        return {}

    info = {
        "liczba_wierszy": df.shape[0],
        "liczba_kolumn": df.shape[1],
        "nazwy_kolumn": df.columns.tolist(),
        "typy_danych": df.dtypes.to_dict(),
        "brakujące_wartości": df.isnull().sum().sum(),
        "duplikaty": df.duplicated().sum()
    }

    # Dodaj informacje o klasach tylko jeśli kolumna 'Class' istnieje
    if 'Class' in df.columns:
        info["liczba_klas"] = df['Class'].nunique()
        info["rozkład_klas"] = df['Class'].value_counts().to_dict()
    else:
        # Sprawdź, czy istnieją kolumny typu Class_1, Class_2, itp.
        class_columns = [col for col in df.columns if col.startswith('Class_')]
        if class_columns:
            info["liczba_klas"] = len(class_columns)
            info["rozkład_klas"] = {i + 1: df[col].sum() for i, col in enumerate(sorted(class_columns))}
        else:
            info["liczba_klas"] = 0
            info["rozkład_klas"] = {}

    return info


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Dzieli zbiór danych na cechy (features) i cel (target).
    W ten sposób rozdzielamy wszystkie kolumny otrzymując:
    X - dane wejściowe
    y - klasę którą model przewiduje

    Args:
        df: DataFrame z danymi

    Returns:
        Tuple zawierający (X, y) gdzie X to cechy, a y to zmienna celu
    """
    if df is None:
        return None, None

    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y


def get_sample_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Zwraca próbkę n pierwszych wierszy zbioru danych.

    Args:
        df: DataFrame z danymi
        n: Liczba wierszy do zwrócenia

    Returns:
        DataFrame z n pierwszymi wierszami
    """
    if df is None:
        return None

    return df.head(n)