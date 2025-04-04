"""
Moduł odpowiedzialny za manipulację danymi: selekcję, transformację, skalowanie, itp.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Union, List, Tuple, Optional


def select_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Wybiera podane kolumny z DataFrame.

    Args:
        df: DataFrame z danymi
        features: Lista nazw kolumn do wybrania

    Returns:
        DataFrame z wybranymi kolumnami
    """
    if df is None:
        return None

    try:
        return df[features].copy()
    except KeyError as e:
        print(f"Błąd: Kolumna {e} nie istnieje w zbiorze danych.")
        return df


def select_rows_by_class(df: pd.DataFrame, class_values: List[int]) -> pd.DataFrame:
    """
    Wybiera wiersze odpowiadające podanym klasom.

    Args:
        df: DataFrame z danymi
        class_values: Lista wartości klasy do wybrania

    Returns:
        DataFrame z wybranymi wierszami
    """
    if df is None:
        return None

    return df[df['Class'].isin(class_values)].copy()


def replace_values(df: pd.DataFrame, column: str, old_value: Union[int, float],
                   new_value: Union[int, float, str, None]) -> pd.DataFrame:
    """
    Zastępuje wartości w podanej kolumnie.

    Args:
        df: DataFrame z danymi
        column: Nazwa kolumny
        old_value: Wartość do zastąpienia
        new_value: Nowa wartość

    Returns:
        DataFrame ze zmienionymi wartościami
    """
    if df is None:
        return None

    result_df = df.copy()
    try:
        result_df[column] = result_df[column].replace(old_value, new_value)
        return result_df
    except KeyError:
        print(f"Błąd: Kolumna {column} nie istnieje w zbiorze danych.")
        return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Obsługuje brakujące wartości w danych.

    Args:
        df: DataFrame z danymi
        strategy: Strategia wypełniania braków ('mean', 'median', 'most_frequent', 'constant')

    Returns:
        DataFrame z uzupełnionymi wartościami
    """
    if df is None:
        return None

    result_df = df.copy()

    # Separowanie kolumn numerycznych i nienerycznych
    numeric_cols = result_df.select_dtypes(include=np.number).columns

    # Wypełnianie brakujących wartości w kolumnach numerycznych
    if numeric_cols.any():
        imputer = SimpleImputer(strategy=strategy)
        result_df[numeric_cols] = imputer.fit_transform(result_df[numeric_cols])

    # Wypełnianie brakujących wartości w kolumnach kategorycznych (np. Class)
    cat_cols = result_df.select_dtypes(exclude=np.number).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            result_df[col] = result_df[col].fillna(result_df[col].mode()[0])

    return result_df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usuwa duplikaty z DataFrame.

    Args:
        df: DataFrame z danymi

    Returns:
        DataFrame bez duplikatów
    """
    if df is None:
        return None

    return df.drop_duplicates().reset_index(drop=True)


def scale_data(df: pd.DataFrame, method: str = 'standard',
               columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Skaluje dane za pomocą wybranej metody.

    Args:
        df: DataFrame z danymi
        method: Metoda skalowania ('standard', 'minmax')
        columns: Lista kolumn do skalowania (domyślnie wszystkie numeryczne)

    Returns:
        DataFrame ze skalowanymi danymi
    """
    if df is None:
        return None

    result_df = df.copy()

    # Jeśli nie podano kolumn, wybierz wszystkie numeryczne
    if columns is None:
        columns = result_df.select_dtypes(include=np.number).columns.tolist()

        # Upewnij się, że nie próbujemy skalować kolumny Class
        if 'Class' in columns:
            columns.remove('Class')

    # Sprawdź, czy podane kolumny istnieją
    valid_columns = [col for col in columns if col in result_df.columns]

    if len(valid_columns) == 0:
        print("Brak poprawnych kolumn do skalowania.")
        return result_df

    # Wybierz odpowiedni skaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        print(f"Nieznana metoda skalowania: {method}. Używam StandardScaler.")
        scaler = StandardScaler()

    # Skalowanie danych
    result_df[valid_columns] = scaler.fit_transform(result_df[valid_columns])

    return result_df


def encode_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Koduje kolumnę Class do formatu one-hot encoding.

    Args:
        df: DataFrame z danymi

    Returns:
        DataFrame z zakodowaną kolumną Class
    """
    if df is None:
        return None

    result_df = df.copy()

    # Sprawdź czy kolumna Class istnieje
    if 'Class' not in result_df.columns:
        print("Błąd: Kolumna 'Class' nie istnieje w zbiorze danych.")
        return result_df

    # Zastosuj one-hot encoding
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(result_df[['Class']])

    # Stwórz DataFrame z zakodowanymi danymi
    encoded_df = pd.DataFrame(
        encoded,
        columns=[f'Class_{i}' for i in range(1, encoded.shape[1] + 1)]
    )

    # Połącz z oryginalnym DataFrame
    result_df = pd.concat([result_df.drop('Class', axis=1).reset_index(drop=True),
                           encoded_df.reset_index(drop=True)], axis=1)

    return result_df


def add_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Dodaje wielomianowe cechy dla wybranych kolumn.

    Args:
        df: DataFrame z danymi
        columns: Lista kolumn do transformacji
        degree: Stopień wielomianu

    Returns:
        DataFrame z dodanymi cechami wielomianowymi
    """
    if df is None:
        return None

    result_df = df.copy()

    # Sprawdź, czy podane kolumny istnieją
    valid_columns = [col for col in columns if col in result_df.columns]

    if len(valid_columns) == 0:
        print("Brak poprawnych kolumn do transformacji.")
        return result_df

    # Dodaj cechy wielomianowe
    for i in range(2, degree + 1):
        for col in valid_columns:
            result_df[f"{col}^{i}"] = result_df[col] ** i

    return result_df