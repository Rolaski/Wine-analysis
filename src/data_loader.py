"""
Moduł odpowiedzialny za wczytywanie i podstawowe operacje na zbiorze danych.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Union, Dict
import sys


def get_app_data_path() -> str:
    """
    Zwraca właściwą ścieżkę do katalogu z danymi w zależności od środowiska.

    Returns:
        Ścieżka do katalogu z danymi
    """
    # Sprawdź czy jest to środowisko Electron
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller bundle
        base_path = sys._MEIPASS
    elif os.environ.get('ELECTRON_RUN_AS_NODE'):
        # Electron environment
        # Sprawdź czy aplikacja jest spakowana
        if os.environ.get('PORTABLE_EXECUTABLE_DIR'):
            # Spakowana aplikacja Electron
            base_path = os.path.join(os.environ.get('PORTABLE_EXECUTABLE_DIR', ''), 'resources', 'streamlit_app')
        else:
            # Development mode - sprawdź różne możliwe lokalizacje
            possible_paths = [
                os.path.join(os.getcwd(), 'streamlit_app'),  # Jeśli CWD to główny katalog projektu
                os.getcwd(),  # Jeśli CWD to katalog streamlit_app
                os.path.dirname(os.path.abspath(__file__)),  # Katalog tego pliku
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_app')
            ]

            base_path = None
            for path in possible_paths:
                if os.path.exists(os.path.join(path, 'data', 'wine.data')):
                    base_path = path
                    break

            if base_path is None:
                # Fallback - użyj katalogu tego pliku
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        # Standardowe środowisko Python/Streamlit
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(base_path, 'data')

    # Upewnij się, że ścieżka istnieje
    if not os.path.exists(data_path):
        # Spróbuj alternatywne lokalizacje
        alternative_paths = [
            os.path.join(os.getcwd(), 'data'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'),
            os.path.join(os.path.dirname(sys.argv[0]), 'data') if sys.argv else None,
        ]

        for alt_path in alternative_paths:
            if alt_path and os.path.exists(alt_path):
                data_path = alt_path
                break

    return data_path


def load_wine_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Wczytuje zbiór danych Wine Dataset i dodaje odpowiednie nagłówki.
    Wywoływana na początku aplikacji.

    Args:
        data_path: Ścieżka do pliku z danymi (opcjonalna)

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

    # Jeśli nie podano ścieżki, znajdź automatycznie
    if data_path is None:
        data_dir = get_app_data_path()
        data_path = os.path.join(data_dir, 'wine.data')

    # Sprawdź możliwe lokalizacje pliku
    possible_files = [
        data_path,
        os.path.join(get_app_data_path(), 'wine.data'),
        os.path.join(os.getcwd(), 'data', 'wine.data'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'wine.data'),
        './data/wine.data',  # Fallback na względną ścieżkę
    ]

    # Spróbuj wczytać z każdej możliwej lokalizacji
    for file_path in possible_files:
        try:
            if file_path and os.path.exists(file_path):
                wine_df = pd.read_csv(file_path, header=None, names=column_names)
                print(f"Wczytano dane z {file_path} pomyślnie.")
                st.info(f"Wczytano dane z: {file_path}")
                return wine_df
        except Exception as e:
            print(f"Błąd podczas wczytywania z {file_path}: {e}")
            continue

    # Jeśli żaden plik nie został znaleziony, wypróbuj dane wbudowane
    print("Nie znaleziono pliku z danymi. Sprawdzam dane wbudowane...")
    return load_embedded_wine_data()


def load_embedded_wine_data() -> pd.DataFrame:
    """
    Wczytuje przykładowe dane wine dataset jako fallback.
    Używane gdy główny plik z danymi nie został znaleziony.

    Returns:
        DataFrame z przykładowymi danymi Wine Dataset (po 5 próbek z każdej klasy)
    """
    # Wyświetl komunikat o problemie
    st.error("⚠️ **Problem z wczytywaniem głównego zbioru danych!**")
    st.warning("""
    **Nie udało się znaleźć pliku wine.data w standardowych lokalizacjach.**

    **Możliwe przyczyny:**
    - Plik nie został skopiowany podczas budowania aplikacji
    - Nieprawidłowe ścieżki w konfiguracji Electron
    - Brakujące uprawnienia dostępu do plików

    **Używam przykładowych danych (15 próbek) do demonstracji funkcjonalności aplikacji.**
    """)

    # Nazwy kolumn
    column_names = [
        'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
    ]

    # Przykładowe dane - po 5 próbek z każdej klasy
    sample_data = [
        # Klasa 1 (5 próbek)
        [1, 14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065],
        [1, 13.20, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050],
        [1, 13.16, 2.36, 2.67, 18.6, 101, 2.8, 3.24, 0.30, 2.81, 5.68, 1.03, 3.17, 1185],
        [1, 14.37, 1.95, 2.50, 16.8, 113, 3.85, 3.49, 0.24, 2.18, 7.80, 0.86, 3.45, 1480],
        [1, 13.24, 2.59, 2.87, 21.0, 118, 2.8, 2.69, 0.39, 1.82, 4.32, 1.04, 2.93, 735],

        # Klasa 2 (5 próbek)
        [2, 12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520],
        [2, 12.33, 1.10, 2.28, 16.0, 101, 2.05, 1.09, 0.63, 0.41, 3.27, 1.25, 1.67, 680],
        [2, 12.64, 1.36, 2.02, 16.8, 100, 2.02, 1.41, 0.53, 0.62, 5.75, 0.98, 1.59, 450],
        [2, 13.67, 1.25, 1.92, 18.0, 94, 2.10, 1.79, 0.32, 0.73, 3.80, 1.23, 2.46, 630],
        [2, 12.37, 1.13, 2.16, 19.0, 87, 3.50, 3.10, 0.19, 1.87, 4.45, 1.22, 2.87, 420],

        # Klasa 3 (5 próbek)
        [3, 12.86, 1.35, 2.32, 18.0, 122, 1.51, 1.25, 0.21, 0.94, 4.10, 0.76, 1.29, 630],
        [3, 12.88, 2.99, 2.40, 20.0, 104, 1.30, 1.22, 0.24, 0.83, 5.40, 0.74, 1.42, 530],
        [3, 12.81, 2.31, 2.40, 24.0, 98, 1.15, 1.09, 0.27, 0.83, 5.70, 0.66, 1.36, 560],
        [3, 12.70, 3.55, 2.36, 21.5, 106, 1.70, 1.20, 0.17, 0.84, 5.00, 0.78, 1.29, 600],
        [3, 12.51, 1.24, 2.25, 17.5, 85, 2.00, 0.58, 0.60, 1.25, 5.45, 0.75, 1.51, 650],
    ]

    # Stwórz DataFrame
    df = pd.DataFrame(sample_data, columns=column_names)

    st.info(f"✅ Załadowano {len(df)} przykładowych próbek (po 5 z każdej klasy) do demonstracji.")

    return df


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