"""
Moduł odpowiedzialny za wczytywanie i podstawowe operacje na zbiorze danych.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Union, Dict
import io


def load_wine_dataset(data_path: str = "./data/wine.data") -> pd.DataFrame:
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


def load_csv_file(uploaded_file, has_header: bool = True, class_column: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Wczytuje dane z przesłanego pliku CSV.

    Args:
        uploaded_file: Przesłany plik (Streamlit UploadedFile)
        has_header: Czy plik ma nagłówki kolumn
        class_column: Nazwa kolumny z klasami (opcjonalne)

    Returns:
        Tuple zawierający (DataFrame, komunikat statusu)
    """
    try:
        # Sprawdź czy to CSV
        if not uploaded_file.name.endswith('.csv'):
            return None, "Błąd: Plik musi mieć rozszerzenie .csv"

        # Wczytaj plik
        if has_header:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, header=None)
            # Nadaj automatyczne nazwy kolumn
            df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]

        # Sprawdź czy DataFrame nie jest pusty
        if df.empty:
            return None, "Błąd: Plik CSV jest pusty"

        # Jeśli podano nazwę kolumny klasy, sprawdź czy istnieje
        if class_column and class_column not in df.columns:
            return None, f"Błąd: Kolumna '{class_column}' nie istnieje w danych"

        # Jeśli nie podano kolumny klasy, ale jest kolumna 'Class', użyj jej
        if not class_column and 'Class' in df.columns:
            class_column = 'Class'

        # Jeśli nie podano kolumny klasy i nie ma 'Class', spróbuj znaleźć kolumnę kategoryczną
        if not class_column:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                class_column = categorical_cols[0]
                message = f"Automatycznie wybrano kolumnę '{class_column}' jako klasę"
            else:
                # Sprawdź czy pierwsza lub ostatnia kolumna może być klasą
                first_col_unique = df.iloc[:, 0].nunique()
                last_col_unique = df.iloc[:, -1].nunique()

                if first_col_unique <= 10 and first_col_unique > 1:  # Prawdopodobnie klasa
                    class_column = df.columns[0]
                    df[class_column] = df[class_column].astype('category')
                    message = f"Automatycznie wybrano pierwszą kolumnę '{class_column}' jako klasę"
                elif last_col_unique <= 10 and last_col_unique > 1:  # Prawdopodobnie klasa
                    class_column = df.columns[-1]
                    df[class_column] = df[class_column].astype('category')
                    message = f"Automatycznie wybrano ostatnią kolumnę '{class_column}' jako klasę"
                else:
                    message = "Uwaga: Nie znaleziono kolumny z klasami. Dane będą traktowane jako nieoznaczone."
        else:
            message = f"Użyto kolumny '{class_column}' jako klasy"

        # Jeśli znaleziono kolumnę klasy, upewnij się że nazywa się 'Class'
        if class_column and class_column != 'Class':
            df = df.rename(columns={class_column: 'Class'})
            message += f" (przemianowano na 'Class')"

        return df, f"Pomyślnie wczytano {len(df)} wierszy i {len(df.columns)} kolumn. " + message

    except Exception as e:
        return None, f"Błąd podczas wczytywania pliku: {str(e)}"


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Waliduje wczytany zbiór danych.

    Args:
        df: DataFrame do walidacji

    Returns:
        Tuple zawierający (czy_poprawny, komunikat)
    """
    if df is None:
        return False, "DataFrame jest pusty"

    if len(df) == 0:
        return False, "DataFrame nie zawiera żadnych wierszy"

    if len(df.columns) < 2:
        return False, "DataFrame musi mieć co najmniej 2 kolumny"

    # Sprawdź czy są jakieś kolumny numeryczne
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return False, "DataFrame musi zawierać co najmniej jedną kolumnę numeryczną"

    # Sprawdź czy kolumna Class ma sensowne wartości
    if 'Class' in df.columns:
        class_unique = df['Class'].nunique()
        if class_unique == 1:
            return False, "Kolumna Class musi zawierać więcej niż jedną unikalną wartość"
        if class_unique > len(df) * 0.5:  # Więcej niż połowa wartości unikalnych
            return False, "Kolumna Class zawiera zbyt wiele unikalnych wartości (może nie być kolumną klas)"

    return True, "Zbiór danych jest poprawny"


def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatycznie wykrywa typy danych w kolumnach.

    Args:
        df: DataFrame do analizy

    Returns:
        Słownik z typami danych dla każdej kolumny
    """
    column_types = {}

    for col in df.columns:
        if col == 'Class':
            column_types[col] = 'categorical'
        elif df[col].dtype in ['int64', 'float64']:
            # Sprawdź czy to może być kategoria mimo że numeryczne
            unique_count = df[col].nunique()
            if unique_count <= 10 and unique_count < len(df) * 0.1:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'numeric'
        elif df[col].dtype == 'object':
            # Spróbuj przekonwertować na numeryczne
            try:
                pd.to_numeric(df[col])
                column_types[col] = 'numeric'
            except:
                column_types[col] = 'categorical'
        else:
            column_types[col] = 'other'

    return column_types


def preprocess_dataset(df: pd.DataFrame, auto_convert: bool = True) -> pd.DataFrame:
    """
    Automatycznie preprocessuje zbiór danych.

    Args:
        df: DataFrame do preprocessingu
        auto_convert: Czy automatycznie konwertować typy danych

    Returns:
        Preprocessowany DataFrame
    """
    result_df = df.copy()

    if auto_convert:
        # Wykryj i konwertuj typy danych
        column_types = detect_data_types(result_df)

        for col, col_type in column_types.items():
            if col_type == 'numeric' and result_df[col].dtype == 'object':
                # Konwertuj na numeryczne
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            elif col_type == 'categorical' and col != 'Class':
                # Konwertuj na kategorie
                result_df[col] = result_df[col].astype('category')

    return result_df


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

    if 'Class' not in df.columns:
        # Jeśli nie ma kolumny Class, zwróć wszystkie dane jako cechy i None jako target
        return df, None

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


def create_data_upload_interface():
    """
    Tworzy interfejs do wczytywania danych z pliku CSV.
    Funkcja pomocnicza dla głównej aplikacji.

    Returns:
        DataFrame z wczytanymi danymi lub None
    """
    st.subheader("📁 Wczytaj własne dane")

    with st.expander("ℹ️ Instrukcja wczytywania danych CSV", expanded=False):
        st.markdown("""
        **Jak przygotować plik CSV:**
        
        1. **Format pliku**: Plik musi mieć rozszerzenie `.csv`
        2. **Kodowanie**: Zalecane UTF-8
        3. **Separator**: Przecinek (,) lub średnik (;)
        4. **Nagłówki**: Pierwszy wiersz powinien zawierać nazwy kolumn
        5. **Kolumna klas**: Powinna zawierać kategorie/etykiety (np. 1, 2, 3 lub A, B, C)
        
        **Przykład struktury:**
        ```
        Cecha1,Cecha2,Cecha3,Class
        1.2,3.4,5.6,A
        2.1,4.3,6.5,B
        ```
        
        **Uwagi:**
        - Aplikacja automatycznie spróbuje wykryć kolumnę z klasami
        - Jeśli nie znajdzie, dane będą traktowane jako nieoznaczone
        - Można ręcznie wyspecyfikować kolumnę z klasami
        """)

    # Upload file widget
    uploaded_file = st.file_uploader(
        "Wybierz plik CSV",
        type=['csv'],
        help="Prześlij plik CSV z danymi do analizy"
    )

    if uploaded_file is not None:
        # Opcje wczytywania
        col1, col2 = st.columns(2)

        with col1:
            has_header = st.checkbox(
                "Plik zawiera nagłówki",
                value=True,
                help="Czy pierwszy wiersz zawiera nazwy kolumn"
            )

        with col2:
            auto_detect = st.checkbox(
                "Automatycznie wykryj kolumnę klas",
                value=True,
                help="Czy automatycznie próbować znaleźć kolumnę z klasami"
            )

        # Podgląd pliku
        if st.button("Podgląd pliku"):
            try:
                # Wczytaj tylko pierwsze kilka wierszy dla podglądu
                preview_df = pd.read_csv(uploaded_file, nrows=5, header=0 if has_header else None)
                st.write("**Podgląd pierwszych 5 wierszy:**")
                st.dataframe(preview_df)

                # Reset file pointer
                uploaded_file.seek(0)

            except Exception as e:
                st.error(f"Błąd podczas podglądu pliku: {str(e)}")

        # Ręczne określenie kolumny klas (jeśli nie auto-detect)
        class_column = None
        if not auto_detect:
            try:
                temp_df = pd.read_csv(uploaded_file, nrows=1, header=0 if has_header else None)
                class_column = st.selectbox(
                    "Wybierz kolumnę z klasami:",
                    ["Brak"] + list(temp_df.columns),
                    help="Wybierz kolumnę zawierającą etykiety klas"
                )
                if class_column == "Brak":
                    class_column = None

                # Reset file pointer
                uploaded_file.seek(0)

            except Exception as e:
                st.error(f"Błąd podczas odczytu kolumn: {str(e)}")

        # Przycisk wczytania
        if st.button("Wczytaj dane", type="primary"):
            with st.spinner("Wczytywanie danych..."):
                df, message = load_csv_file(uploaded_file, has_header, class_column)

                if df is not None:
                    # Walidacja
                    is_valid, validation_message = validate_dataset(df)

                    if is_valid:
                        # Preprocessing
                        df = preprocess_dataset(df)

                        st.success(message)
                        st.info(validation_message)

                        # Pokaż podstawowe informacje
                        info = get_dataset_info(df)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Wiersze", info["liczba_wierszy"])
                        with col2:
                            st.metric("Kolumny", info["liczba_kolumn"])
                        with col3:
                            if info["liczba_klas"] > 0:
                                st.metric("Klasy", info["liczba_klas"])
                            else:
                                st.metric("Klasy", "Brak")

                        # Podgląd danych
                        st.write("**Podgląd wczytanych danych:**")
                        st.dataframe(df.head())

                        return df
                    else:
                        st.error(f"Walidacja nie powiodła się: {validation_message}")
                else:
                    st.error(message)

    return None


def get_sample_wine_data() -> pd.DataFrame:
    """
    Zwraca przykładowe dane Wine jako awaryjne rozwiązanie.

    Returns:
        DataFrame z przykładowymi danymi Wine
    """
    from src.utils import get_sample_wine_data
    return get_sample_wine_data()