"""
Moduł odpowiedzialny za wizualizację danych.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union
import io
from matplotlib.figure import Figure

# Ustawienie stylu dla wykresów
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})


def create_histogram(df: pd.DataFrame, column: str, bins: int = 10,
                     by_class: bool = False) -> Figure:
    """
    Tworzy histogram dla wybranej kolumny.

    Args:
        df: DataFrame z danymi
        column: Nazwa kolumny do wizualizacji
        bins: Liczba przedziałów histogramu
        by_class: Czy grupować według klasy

    Returns:
        Obiekt Figure z wykresem
    """
    if df is None or column not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Brak danych dla kolumny {column}",
                horizontalalignment='center', verticalalignment='center')
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))

    if by_class and 'Class' in df.columns:
        for class_value in sorted(df['Class'].unique()):
            subset = df[df['Class'] == class_value]
            sns.histplot(subset[column], bins=bins, kde=True, alpha=0.6,
                         label=f'Klasa {class_value}', ax=ax)
        plt.legend(title='Klasa')
    else:
        sns.histplot(df[column], bins=bins, kde=True, ax=ax)

    plt.title(f'Histogram dla cechy: {column}')
    plt.xlabel(column)
    plt.ylabel('Liczność')
    plt.tight_layout()

    return fig


def create_boxplot(df: pd.DataFrame, column: Optional[str] = None,
                   by_class: bool = True) -> Figure:
    """
    Tworzy wykres pudełkowy dla wybranej kolumny.

    Args:
        df: DataFrame z danymi
        column: Nazwa kolumny do wizualizacji (jeśli None, to dla wszystkich numerycznych)
        by_class: Czy grupować według klasy

    Returns:
        Obiekt Figure z wykresem
    """
    if df is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Brak danych",
                horizontalalignment='center', verticalalignment='center')
        return fig

    # Wybierz kolumny do wizualizacji
    if column is not None:
        if column not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Kolumna {column} nie istnieje w zbiorze danych",
                    horizontalalignment='center', verticalalignment='center')
            return fig
        columns_to_plot = [column]
    else:
        # Wybierz wszystkie kolumny numeryczne oprócz 'Class'
        columns_to_plot = df.select_dtypes(include=np.number).columns.tolist()
        if 'Class' in columns_to_plot:
            columns_to_plot.remove('Class')

    fig, ax = plt.subplots(figsize=(12, 8))

    if by_class and 'Class' in df.columns and len(columns_to_plot) == 1:
        # Boxplot dla jednej kolumny z grupowaniem według klasy
        sns.boxplot(x='Class', y=columns_to_plot[0], data=df, ax=ax)
        plt.title(f'Wykres pudełkowy dla cechy: {columns_to_plot[0]} według klasy')
        plt.xlabel('Klasa')
        plt.ylabel(columns_to_plot[0])
    elif len(columns_to_plot) == 1:
        # Boxplot dla jednej kolumny bez grupowania
        sns.boxplot(y=columns_to_plot[0], data=df, ax=ax)
        plt.title(f'Wykres pudełkowy dla cechy: {columns_to_plot[0]}')
        plt.ylabel(columns_to_plot[0])
    else:
        # Boxplot dla wielu kolumn
        melted_df = pd.melt(df[columns_to_plot])
        sns.boxplot(x='variable', y='value', data=melted_df, ax=ax)
        plt.title('Wykres pudełkowy dla wszystkich cech')
        plt.xlabel('Cecha')
        plt.ylabel('Wartość')
        plt.xticks(rotation=90)

    plt.tight_layout()
    return fig


def create_scatter_plot(df: pd.DataFrame, x: str, y: str,
                        color_by_class: bool = True) -> Figure:
    """
    Tworzy wykres rozproszenia dla dwóch kolumn.

    Args:
        df: DataFrame z danymi
        x: Nazwa kolumny dla osi X
        y: Nazwa kolumny dla osi Y
        color_by_class: Czy kolorować punkty według klasy

    Returns:
        Obiekt Figure z wykresem
    """
    if df is None or x not in df.columns or y not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Brak danych dla kolumn {x} i/lub {y}",
                horizontalalignment='center', verticalalignment='center')
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))

    if color_by_class and 'Class' in df.columns:
        for class_value in sorted(df['Class'].unique()):
            subset = df[df['Class'] == class_value]
            ax.scatter(subset[x], subset[y], alpha=0.7, label=f'Klasa {class_value}')
        plt.legend(title='Klasa')
    else:
        ax.scatter(df[x], df[y], alpha=0.7)

    plt.title(f'Wykres rozproszenia: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()

    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None,
                               method: str = 'pearson', include_class: bool = False) -> Figure:
    """
    Tworzy macierz korelacji jako wykres cieplny.

    Args:
        df: DataFrame z danymi
        columns: Lista kolumn do analizy (domyślnie wszystkie numeryczne)
        method: Metoda korelacji ('pearson', 'spearman', 'kendall')
        include_class: Czy uwzględnić kolumnę 'Class'

    Returns:
        Obiekt Figure z wykresem
    """
    if df is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Brak danych",
                horizontalalignment='center', verticalalignment='center')
        return fig

    # Wybierz kolumny do analizy
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        if not include_class and 'Class' in columns:
            columns.remove('Class')
    else:
        columns = [col for col in columns if col in df.columns]

    if len(columns) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Brak odpowiednich kolumn do analizy",
                horizontalalignment='center', verticalalignment='center')
        return fig

    # Oblicz macierz korelacji
    corr_matrix = df[columns].corr(method=method)

    # Stwórz wykres
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)

    plt.title(f'Macierz korelacji ({method})')
    plt.tight_layout()

    return fig


def create_pairplot(df: pd.DataFrame, columns: Optional[List[str]] = None,
                    hue: str = 'Class', max_cols: int = 5) -> Figure:
    """
    Tworzy wykres par cech.

    Args:
        df: DataFrame z danymi
        columns: Lista kolumn do analizy (domyślnie wybrane numeryczne)
        hue: Kolumna do kolorowania (domyślnie 'Class')
        max_cols: Maksymalna liczba kolumn do uwzględnienia

    Returns:
        Obiekt Figure z wykresem
    """
    if df is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Brak danych",
                horizontalalignment='center', verticalalignment='center')
        return fig

    # Sprawdź, czy kolumna hue istnieje
    if hue not in df.columns:
        hue = None

    # Wybierz kolumny do analizy
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        if hue in columns:
            columns.remove(hue)
    else:
        columns = [col for col in columns if col in df.columns]

    # Ograniczenie liczby kolumn
    if len(columns) > max_cols:
        # Wybierz pierwsze max_cols kolumn
        columns = columns[:max_cols]

    if len(columns) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Za mało odpowiednich kolumn do analizy",
                horizontalalignment='center', verticalalignment='center')
        return fig

    # Utwórz pairplot
    g = sns.pairplot(df[columns + ([hue] if hue else [])],
                     hue=hue,
                     height=2.5,
                     diag_kind='kde')

    g.fig.suptitle('Wykres par cech', y=1.02)
    plt.tight_layout()

    return g.fig


def create_feature_importance(features: List[str], importance: List[float]) -> Figure:
    """
    Tworzy wykres ważności cech.

    Args:
        features: Lista nazw cech
        importance: Lista wartości ważności cech

    Returns:
        Obiekt Figure z wykresem
    """
    if not features or not importance or len(features) != len(importance):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Niepoprawne dane do wizualizacji ważności cech",
                horizontalalignment='center', verticalalignment='center')
        return fig

    # Sortowanie cech według ważności
    sorted_idx = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Stwórz wykres poziomych pasków
    ax.barh(range(len(sorted_features)), sorted_importance, align='center')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()  # Najważniejsze cechy na górze

    plt.title('Ważność cech')
    plt.xlabel('Ważność')
    plt.tight_layout()

    return fig


def create_class_distribution(df: pd.DataFrame) -> Figure:
    """
    Tworzy wykres rozkładu klas.

    Args:
        df: DataFrame z danymi

    Returns:
        Obiekt Figure z wykresem
    """
    if df is None or 'Class' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Brak danych o klasach",
                horizontalalignment='center', verticalalignment='center')
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))

    # Policz wystąpienia każdej klasy
    class_counts = df['Class'].value_counts().sort_index()

    # Stwórz wykres
    ax.bar(class_counts.index.astype(str), class_counts.values)

    plt.title('Rozkład klas')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba próbek')

    # Dodaj etykiety z liczbą próbek
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 5, str(v), ha='center')

    plt.tight_layout()

    return fig


def create_3d_scatter(df: pd.DataFrame, x: str, y: str, z: str,
                      color_by_class: bool = True) -> Figure:
    """
    Tworzy trójwymiarowy wykres rozproszenia.

    Args:
        df: DataFrame z danymi
        x: Nazwa kolumny dla osi X
        y: Nazwa kolumny dla osi Y
        z: Nazwa kolumny dla osi Z
        color_by_class: Czy kolorować punkty według klasy

    Returns:
        Obiekt Figure z wykresem
    """
    if df is None or x not in df.columns or y not in df.columns or z not in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Brak danych dla kolumn {x}, {y} i/lub {z}",
                horizontalalignment='center', verticalalignment='center')
        return fig

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if color_by_class and 'Class' in df.columns:
        classes = sorted(df['Class'].unique())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

        for cls, color in zip(classes, colors):
            subset = df[df['Class'] == cls]
            ax.scatter(subset[x], subset[y], subset[z],
                       color=color, label=f'Klasa {cls}', alpha=0.7)

        plt.legend(title='Klasa')
    else:
        ax.scatter(df[x], df[y], df[z], alpha=0.7)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.title(f'Wykres 3D: {x} vs {y} vs {z}')

    return fig


def create_parallel_coordinates(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                class_column: str = 'Class', max_cols: int = 8) -> Figure:
    """
    Tworzy wykres współrzędnych równoległych.

    Args:
        df: DataFrame z danymi
        columns: Lista kolumn do analizy (domyślnie wybrane numeryczne)
        class_column: Nazwa kolumny z klasami
        max_cols: Maksymalna liczba kolumn do uwzględnienia

    Returns:
        Obiekt Figure z wykresem
    """
    if df is None or class_column not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Brak danych lub kolumny {class_column}",
                horizontalalignment='center', verticalalignment='center')
        return fig

    # Wybierz kolumny do analizy
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        if class_column in columns:
            columns.remove(class_column)
    else:
        columns = [col for col in columns if col in df.columns]

    # Ograniczenie liczby kolumn
    if len(columns) > max_cols:
        # Wybierz pierwsze max_cols kolumn
        columns = columns[:max_cols]

    if len(columns) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Za mało odpowiednich kolumn do analizy",
                horizontalalignment='center', verticalalignment='center')
        return fig

    # Przygotuj dane do wykresu
    plot_df = df[columns + [class_column]].copy()

    # Skalowanie danych do zakresu [0, 1] dla lepszej wizualizacji
    for col in columns:
        plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min())

    fig, ax = plt.subplots(figsize=(12, 8))

    # Stwórz wykres
    pd.plotting.parallel_coordinates(plot_df, class_column, ax=ax)

    plt.title('Wykres współrzędnych równoległych')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig