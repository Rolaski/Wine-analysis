"""
Moduł odpowiedzialny za analizę statystyczną danych.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Union, List, Optional


def calculate_basic_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Oblicza podstawowe statystyki dla danych.

    Args:
        df: DataFrame z danymi
        columns: Lista kolumn do analizy (domyślnie wszystkie numeryczne)

    Returns:
        DataFrame ze statystykami
    """
    if df is None:
        return None

    # Jeśli nie podano kolumn, wybierz wszystkie numeryczne
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    # Sprawdź, czy podane kolumny istnieją
    valid_columns = [col for col in columns if col in df.columns]

    if len(valid_columns) == 0:
        print("Brak poprawnych kolumn do analizy.")
        return pd.DataFrame()

    # Oblicz statystyki
    stats_df = pd.DataFrame({
        'minimum': df[valid_columns].min(),
        'maksimum': df[valid_columns].max(),
        'średnia': df[valid_columns].mean(),
        'mediana': df[valid_columns].median(),
        'odchylenie_std': df[valid_columns].std(),
        'wariancja': df[valid_columns].var(),
        'skośność': df[valid_columns].skew(),
        'kurtoza': df[valid_columns].kurtosis()
    })

    # Dodaj modę (może nie być jednoznaczna)
    mode_values = {}
    for col in valid_columns:
        mode_result = df[col].mode()
        mode_values[col] = mode_result[0] if not mode_result.empty else np.nan

    stats_df['moda'] = pd.Series(mode_values)

    return stats_df


def calculate_quartiles(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Oblicza kwartyle dla danych.

    Args:
        df: DataFrame z danymi
        columns: Lista kolumn do analizy (domyślnie wszystkie numeryczne)

    Returns:
        DataFrame z kwartylami
    """
    if df is None:
        return None

    # Jeśli nie podano kolumn, wybierz wszystkie numeryczne
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    # Sprawdź, czy podane kolumny istnieją
    valid_columns = [col for col in columns if col in df.columns]

    if len(valid_columns) == 0:
        print("Brak poprawnych kolumn do analizy.")
        return pd.DataFrame()

    # Oblicz kwartyle i percentyle
    quartiles_df = pd.DataFrame({
        'q25': df[valid_columns].quantile(0.25),
        'q50': df[valid_columns].quantile(0.50),
        'q75': df[valid_columns].quantile(0.75),
        'iqr': df[valid_columns].quantile(0.75) - df[valid_columns].quantile(0.25),
        'p10': df[valid_columns].quantile(0.10),
        'p90': df[valid_columns].quantile(0.90)
    })

    return quartiles_df


def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson',
                                 columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Oblicza macierz korelacji dla danych.

    Args:
        df: DataFrame z danymi
        method: Metoda korelacji ('pearson', 'spearman', 'kendall')
        columns: Lista kolumn do analizy (domyślnie wszystkie numeryczne)

    Returns:
        DataFrame z macierzą korelacji
    """
    if df is None:
        return None

    # Jeśli nie podano kolumn, wybierz wszystkie numeryczne
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    # Sprawdź, czy podane kolumny istnieją
    valid_columns = [col for col in columns if col in df.columns]

    if len(valid_columns) == 0:
        print("Brak poprawnych kolumn do analizy.")
        return pd.DataFrame()

    # Oblicz macierz korelacji
    corr_matrix = df[valid_columns].corr(method=method)

    return corr_matrix


def find_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.7,
                                    method: str = 'pearson') -> pd.DataFrame:
    """
    Znajduje pary cech o wysokiej korelacji.

    Args:
        df: DataFrame z danymi
        threshold: Próg korelacji
        method: Metoda korelacji ('pearson', 'spearman', 'kendall')

    Returns:
        DataFrame z parami cech o wysokiej korelacji
    """
    if df is None:
        return None

    # Oblicz macierz korelacji
    corr_matrix = calculate_correlation_matrix(df, method)

    if corr_matrix is None or corr_matrix.empty:
        return pd.DataFrame()

    # Znajdź pary cech o wysokiej korelacji
    corr_pairs = []

    # Przeszukaj górną trójkątną macierz korelacji
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value >= threshold:
                corr_pairs.append({
                    'cecha1': corr_matrix.columns[i],
                    'cecha2': corr_matrix.columns[j],
                    'korelacja': corr_matrix.iloc[i, j],
                    'korelacja_abs': corr_value
                })

    # Utwórz DataFrame z wynikami
    if corr_pairs:
        result_df = pd.DataFrame(corr_pairs)
        return result_df.sort_values('korelacja_abs', ascending=False)
    else:
        return pd.DataFrame(columns=['cecha1', 'cecha2', 'korelacja', 'korelacja_abs'])


def calculate_class_stats(df: pd.DataFrame,
                          feature_columns: Optional[List[str]] = None) -> Dict[int, pd.DataFrame]:
    """
    Oblicza statystyki dla każdej klasy osobno.

    Args:
        df: DataFrame z danymi
        feature_columns: Lista kolumn do analizy (domyślnie wszystkie numeryczne z wyjątkiem 'Class')

    Returns:
        Słownik ze statystykami dla każdej klasy
    """
    if df is None or 'Class' not in df.columns:
        return {}

    # Jeśli nie podano kolumn, wybierz wszystkie numeryczne oprócz 'Class'
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=np.number).columns.tolist()
        if 'Class' in feature_columns:
            feature_columns.remove('Class')

    # Sprawdź, czy podane kolumny istnieją
    valid_columns = [col for col in feature_columns if col in df.columns]

    if len(valid_columns) == 0:
        print("Brak poprawnych kolumn do analizy.")
        return {}

    # Oblicz statystyki dla każdej klasy
    classes = df['Class'].unique()
    class_stats = {}

    for class_value in classes:
        class_df = df[df['Class'] == class_value]
        class_stats[class_value] = calculate_basic_stats(class_df, valid_columns)

    return class_stats


def test_normality(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Przeprowadza test normalności Shapiro-Wilka dla podanych kolumn.

    Args:
        df: DataFrame z danymi
        columns: Lista kolumn do analizy (domyślnie wszystkie numeryczne)

    Returns:
        DataFrame z wynikami testu
    """
    if df is None:
        return None

    # Jeśli nie podano kolumn, wybierz wszystkie numeryczne
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    # Sprawdź, czy podane kolumny istnieją
    valid_columns = [col for col in columns if col in df.columns]

    if len(valid_columns) == 0:
        print("Brak poprawnych kolumn do analizy.")
        return pd.DataFrame()

    # Przeprowadź test normalności
    results = []

    for col in valid_columns:
        # Shapiro-Wilk test działa dla próbek do 5000 elementów
        sample = df[col].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)

        stat, p_value = stats.shapiro(sample)

        results.append({
            'cecha': col,
            'statystyka': stat,
            'p_value': p_value,
            'rozkład_normalny': p_value > 0.05
        })

    return pd.DataFrame(results)


def calculate_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = 'iqr') -> Dict[str, pd.DataFrame]:
    """
    Znajduje wartości odstające w danych za pomocą wybranej metody.

    Args:
        df: DataFrame z danymi
        columns: Lista kolumn do analizy (domyślnie wszystkie numeryczne)
        method: Metoda wykrywania wartości odstających ('iqr', 'zscore')

    Returns:
        Słownik z DataFrames zawierającymi wartości odstające dla każdej kolumny
    """
    if df is None:
        return {}

    # Jeśli nie podano kolumn, wybierz wszystkie numeryczne
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    # Sprawdź, czy podane kolumny istnieją
    valid_columns = [col for col in columns if col in df.columns]

    if len(valid_columns) == 0:
        print("Brak poprawnych kolumn do analizy.")
        return {}

    outliers = {}

    for col in valid_columns:
        if method == 'iqr':
            # Metoda IQR (Interquartile Range)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Znajdź wartości odstające
            outliers_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        elif method == 'zscore':
            # Metoda Z-score
            z_scores = np.abs(stats.zscore(df[col]))
            # Za wartości odstające uznajemy te, które mają |z| > 3
            outliers_df = df[z_scores > 3]
        else:
            print(f"Nieznana metoda wykrywania wartości odstających: {method}")
            continue

        outliers[col] = outliers_df

    return outliers