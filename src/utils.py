"""
Moduł z funkcjami pomocniczymi dla aplikacji Wine Analysis.
"""

import pandas as pd
import numpy as np
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def parse_row_ranges(range_string: str, max_rows: int) -> List[int]:
    """
    Parsuje string z zakresami wierszy do listy indeksów.

    Args:
        range_string: String z zakresami np. "1-5,8,10-12"
        max_rows: Maksymalna liczba wierszy w DataFrame

    Returns:
        Lista indeksów wierszy do usunięcia

    Raises:
        ValueError: Jeśli format jest niepoprawny
    """
    if not range_string.strip():
        return []

    # Usuń spacje i podziel na części
    parts = [part.strip() for part in range_string.split(',')]
    indices = []

    for part in parts:
        if not part:
            continue

        # Sprawdź czy to zakres (zawiera myślnik)
        if '-' in part:
            # Parsuj zakres
            try:
                start_str, end_str = part.split('-', 1)
                start = int(start_str.strip())
                end = int(end_str.strip())

                # Walidacja zakresu
                if start < 0 or end < 0:
                    raise ValueError(f"Indeksy nie mogą być ujemne: {part}")
                if start >= max_rows or end >= max_rows:
                    raise ValueError(f"Indeks poza zakresem (max {max_rows-1}): {part}")
                if start > end:
                    raise ValueError(f"Początek zakresu większy od końca: {part}")

                # Dodaj wszystkie indeksy z zakresu
                indices.extend(range(start, end + 1))

            except ValueError as e:
                if "invalid literal for int()" in str(e):
                    raise ValueError(f"Niepoprawny format zakresu: {part}")
                raise e
        else:
            # Parsuj pojedynczy indeks
            try:
                index = int(part.strip())

                # Walidacja indeksu
                if index < 0:
                    raise ValueError(f"Indeks nie może być ujemny: {index}")
                if index >= max_rows:
                    raise ValueError(f"Indeks poza zakresem (max {max_rows-1}): {index}")

                indices.append(index)

            except ValueError as e:
                if "invalid literal for int()" in str(e):
                    raise ValueError(f"Niepoprawny format indeksu: {part}")
                raise e

    # Usuń duplikaty i posortuj
    return sorted(list(set(indices)))


def parse_value_range(range_string: str) -> Tuple[float, float]:
    """
    Parsuje string z zakresem wartości do krotki (min, max).

    Args:
        range_string: String z zakresem np. "0.5-0.7"

    Returns:
        Krotka (min_val, max_val)

    Raises:
        ValueError: Jeśli format jest niepoprawny
    """
    if not range_string.strip():
        raise ValueError("Pusty string zakresu")

    # Usuń spacje
    range_string = range_string.strip()

    # Sprawdź czy zawiera myślnik
    if '-' not in range_string:
        raise ValueError("Zakres musi zawierać myślnik (np. '0.5-0.7')")

    # Podziel na części
    parts = range_string.split('-')

    # Obsłuż przypadek liczb ujemnych
    if len(parts) == 3 and parts[0] == '':
        # Przypadek: "-1.5-0.5" -> ['', '1.5', '0.5']
        min_str = '-' + parts[1]
        max_str = parts[2]
    elif len(parts) == 4 and parts[0] == '' and parts[2] == '':
        # Przypadek: "-1.5--0.5" -> ['', '1.5', '', '0.5']
        min_str = '-' + parts[1]
        max_str = '-' + parts[3]
    elif len(parts) == 2:
        # Normalny przypadek: "0.5-0.7" -> ['0.5', '0.7']
        min_str = parts[0]
        max_str = parts[1]
    else:
        raise ValueError(f"Niepoprawny format zakresu: {range_string}")

    try:
        min_val = float(min_str.strip())
        max_val = float(max_str.strip())
    except ValueError:
        raise ValueError(f"Niepoprawne wartości liczbowe w zakresie: {range_string}")

    # Walidacja zakresu
    if min_val > max_val:
        raise ValueError(f"Minimum większe od maksimum: {min_val} > {max_val}")

    return min_val, max_val


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Zwraca słownik z podziałem kolumn według typu danych.

    Args:
        df: DataFrame z danymi

    Returns:
        Słownik z nazwami kolumn pogrupowanymi według typu
    """
    if df is None:
        return {}

    column_types = {
        'numeric': df.select_dtypes(include=np.number).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'binary': [col for col in df.columns if df[col].nunique() == 2],
        'datetime': df.select_dtypes(include=np.datetime64).columns.tolist(),
        'all': df.columns.tolist()
    }

    return column_types


def create_feature_column_pairs(columns: List[str], max_pairs: int = 15) -> List[Tuple[str, str]]:
    """
    Tworzy pary kolumn do analizy.

    Args:
        columns: Lista nazw kolumn
        max_pairs: Maksymalna liczba par

    Returns:
        Lista krotek par kolumn
    """
    if not columns or len(columns) < 2:
        return []

    import itertools

    # Generuj wszystkie możliwe kombinacje par kolumn
    all_pairs = list(itertools.combinations(columns, 2))

    # Ograniczenie liczby par
    if len(all_pairs) > max_pairs:
        # Wybierz max_pairs par losowo
        import random
        random.seed(42)  # dla powtarzalności
        return random.sample(all_pairs, max_pairs)

    return all_pairs


def convert_fig_to_html(fig: plt.Figure) -> str:
    """
    Konwertuje obiekt Figure na kod HTML.

    Args:
        fig: Obiekt Figure z matplotlib

    Returns:
        String HTML z zakodowanym obrazem
    """
    if fig is None:
        return "<p>Brak wykresu do wyświetlenia.</p>"

    # Zapisz wykres do bufora
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Konwertuj do base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    # Zamknij wykres, aby zwolnić pamięć
    plt.close(fig)

    # Zwróć kod HTML
    return f'<img src="data:image/png;base64,{img_str}" alt="Wykres">'


def save_model(model: Any, filename: str, directory: str = './models') -> str:
    """
    Zapisuje model do pliku.

    Args:
        model: Obiekt modelu do zapisania
        filename: Nazwa pliku
        directory: Katalog docelowy

    Returns:
        Ścieżka do zapisanego pliku
    """
    # Utwórz katalog, jeśli nie istnieje
    os.makedirs(directory, exist_ok=True)

    # Pełna ścieżka
    path = os.path.join(directory, filename)

    # Zapisz model
    with open(path, 'wb') as f:
        pickle.dump(model, f)

    return path


def load_model(filename: str, directory: str = './models') -> Any:
    """
    Wczytuje model z pliku.

    Args:
        filename: Nazwa pliku
        directory: Katalog źródłowy

    Returns:
        Wczytany obiekt modelu
    """
    # Pełna ścieżka
    path = os.path.join(directory, filename)

    # Sprawdź, czy plik istnieje
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plik {path} nie istnieje.")

    # Wczytaj model
    with open(path, 'rb') as f:
        model = pickle.load(f)

    return model


def format_classification_report(report: Dict) -> pd.DataFrame:
    """
    Formatuje raport klasyfikacji do postaci DataFrame.

    Args:
        report: Słownik z raportem klasyfikacji

    Returns:
        DataFrame z sformatowanym raportem
    """
    if not report:
        return pd.DataFrame()

    # Usuń klucze, które nie są klasami
    metrics_to_remove = ['accuracy', 'macro avg', 'weighted avg']
    classes_report = {k: v for k, v in report.items() if k not in metrics_to_remove}

    # Utwórz DataFrame
    df = pd.DataFrame(classes_report).T.round(3)

    # Dodaj dodatkowe metryki
    if 'macro avg' in report:
        df.loc['średnia makro'] = pd.Series(report['macro avg']).round(3)
    if 'weighted avg' in report:
        df.loc['średnia ważona'] = pd.Series(report['weighted avg']).round(3)

    # Dodaj kolumnę accuracy
    if 'accuracy' in report:
        df['accuracy'] = pd.NA
        df.loc['ogółem', 'accuracy'] = round(report['accuracy'], 3)

    return df


def format_confusion_matrix(matrix: List[List[int]], classes: List[str]) -> pd.DataFrame:
    """
    Formatuje macierz pomyłek do postaci DataFrame.

    Args:
        matrix: Macierz pomyłek jako lista list
        classes: Lista etykiet klas

    Returns:
        DataFrame z sformatowaną macierzą pomyłek
    """
    if not matrix:
        return pd.DataFrame()

    # Utwórz DataFrame
    df = pd.DataFrame(matrix, index=classes, columns=classes)

    # Dodaj etykiety
    df.index.name = 'Prawdziwa klasa'
    df.columns.name = 'Predykcja'

    return df


def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Oblicza metryki dla klasyfikacji binarnej.

    Args:
        y_true: Prawdziwe etykiety
        y_pred: Przewidziane etykiety

    Returns:
        Słownik z metrykami
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef
    )

    # Przekształć do tablic numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Oblicz metryki
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1-score': f1_score(y_true, y_pred, average='binary'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    # Oblicz ROC AUC, jeśli to możliwe
    try:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['ROC AUC'] = None

    return metrics


def generate_model_parameter_description(model_type: str) -> Dict:
    """
    Generuje opisy parametrów dla wybranego typu modelu.

    Args:
        model_type: Typ modelu

    Returns:
        Słownik z opisami parametrów
    """
    if model_type == 'knn':
        return {
            'n_neighbors': {
                'name': 'Liczba sąsiadów',
                'description': 'Liczba sąsiadów do uwzględnienia przy klasyfikacji.',
                'default': 5,
                'min': 1,
                'max': 20
            },
            'weights': {
                'name': 'Wagi',
                'description': 'Funkcja wag używana w predykcji.',
                'options': ['uniform', 'distance'],
                'default': 'uniform'
            }
        }

    elif model_type == 'svm':
        return {
            'C': {
                'name': 'Parametr regularyzacji C',
                'description': 'Parametr kary błędu. Mniejsze wartości oznaczają silniejszą regularyzację.',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'step': 0.1
            },
            'kernel': {
                'name': 'Funkcja jądra',
                'description': 'Określa typ funkcji jądra.',
                'options': ['linear', 'poly', 'rbf', 'sigmoid'],
                'default': 'rbf'
            },
            'gamma': {
                'name': 'Parametr gamma',
                'description': 'Współczynnik jądra dla "rbf", "poly" i "sigmoid".',
                'options': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001],
                'default': 'scale'
            }
        }

    elif model_type == 'rf':
        return {
            'n_estimators': {
                'name': 'Liczba drzew',
                'description': 'Liczba drzew w lesie.',
                'default': 100,
                'min': 10,
                'max': 500,
                'step': 10
            },
            'max_depth': {
                'name': 'Maksymalna głębokość',
                'description': 'Maksymalna głębokość drzewa. None oznacza brak ograniczenia.',
                'default': None,
                'options': [None, 5, 10, 15, 20, 25, 30]
            },
            'min_samples_split': {
                'name': 'Min. próbek do podziału',
                'description': 'Minimalna liczba próbek wymagana do podziału węzła.',
                'default': 2,
                'min': 2,
                'max': 20
            }
        }

    elif model_type == 'kmeans':
        return {
            'n_clusters': {
                'name': 'Liczba klastrów',
                'description': 'Liczba klastrów do wygenerowania.',
                'default': 3,
                'min': 2,
                'max': 10
            },
            'init': {
                'name': 'Metoda inicjalizacji',
                'description': 'Metoda inicjalizacji centroidów klastrów.',
                'options': ['k-means++', 'random'],
                'default': 'k-means++'
            }
        }

    elif model_type == 'dbscan':
        return {
            'eps': {
                'name': 'Epsilon',
                'description': 'Maksymalna odległość między próbkami.',
                'default': 0.5,
                'min': 0.1,
                'max': 2.0,
                'step': 0.1
            },
            'min_samples': {
                'name': 'Min. liczba próbek',
                'description': 'Minimalna liczba próbek w regionie dla rdzenia.',
                'default': 5,
                'min': 2,
                'max': 20
            }
        }

    elif model_type == 'apriori':
        return {
            'min_support': {
                'name': 'Min. wsparcie',
                'description': 'Minimalne wsparcie dla zbiorów elementów.',
                'default': 0.1,
                'min': 0.01,
                'max': 0.5,
                'step': 0.01
            },
            'min_confidence': {
                'name': 'Min. pewność',
                'description': 'Minimalna pewność dla reguł.',
                'default': 0.7,
                'min': 0.5,
                'max': 1.0,
                'step': 0.05
            },
            'metric': {
                'name': 'Metryka',
                'description': 'Metryka do oceny reguł.',
                'options': ['confidence', 'lift', 'leverage', 'conviction'],
                'default': 'confidence'
            }
        }

    else:
        return {}


def get_sample_wine_data() -> pd.DataFrame:
    """
    Zwraca przykładowe dane Wine jako awaryjne rozwiązanie.

    Returns:
        DataFrame z przykładowymi danymi Wine
    """
    data = {
        'Class': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        'Alcohol': [14.23, 13.20, 13.16, 14.37, 13.24, 12.37, 12.33, 12.64, 13.67, 12.37, 12.86, 12.88, 13.05, 13.50,
                    13.05],
        'Malic acid': [1.71, 1.78, 2.36, 1.95, 2.59, 1.63, 0.99, 1.36, 1.25, 1.17, 1.35, 2.99, 1.65, 1.81, 2.36],
        'Ash': [2.43, 2.14, 2.67, 2.50, 2.87, 2.30, 1.95, 2.19, 1.92, 1.92, 2.32, 2.40, 2.55, 1.60, 2.35],
        'Alcalinity of ash': [15.6, 11.2, 18.6, 16.8, 21.0, 19.7, 14.8, 11.5, 18.8, 19.6, 18.0, 15.2, 15.0, 17.0, 16.0],
        'Magnesium': [127, 100, 101, 113, 118, 101, 129, 114, 94, 102, 112, 108, 113, 114, 97],
        'Total phenols': [2.80, 2.65, 2.80, 3.85, 2.80, 3.10, 2.74, 2.88, 2.10, 2.89, 1.98, 1.82, 2.26, 1.87, 1.85],
        'Flavanoids': [3.06, 2.76, 3.24, 3.49, 2.69, 3.22, 2.50, 2.54, 1.82, 2.23, 2.02, 1.43, 2.29, 1.69, 1.64],
        'Nonflavanoid phenols': [0.28, 0.26, 0.30, 0.34, 0.39, 0.31, 0.30, 0.25, 0.32, 0.33, 0.41, 0.99, 0.29, 0.41,
                                 0.27],
        'Proanthocyanins': [2.29, 1.28, 2.81, 1.97, 1.82, 1.46, 1.06, 0.78, 1.35, 1.55, 1.41, 5.75, 1.56, 1.35, 1.25],
        'Color intensity': [5.64, 4.38, 5.68, 6.2, 4.32, 5.48, 3.84, 4.90, 2.76, 4.10, 3.52, 3.00, 4.35, 5.43, 3.03],
        'Hue': [1.04, 1.05, 1.03, 1.07, 1.04, 0.92, 1.10, 1.04, 0.90, 0.91, 1.03, 0.84, 1.02, 1.15, 0.96],
        'OD280/OD315 of diluted wines': [3.92, 3.40, 3.17, 3.33, 2.93, 2.35, 2.56, 2.77, 2.56, 3.20, 3.40, 2.79, 3.58,
                                         3.68, 2.78],
        'Proline': [1065, 1050, 1185, 1080, 735, 870, 830, 1150, 725, 1060, 985, 885, 1095, 780, 680]
    }

    return pd.DataFrame(data)