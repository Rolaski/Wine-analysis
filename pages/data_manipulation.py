"""
Moduł odpowiedzialny za stronę manipulacji danymi w aplikacji Wine Dataset Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Import modułów własnych
from src.data_loader import get_dataset_info
from src.data_manipulator import (
    select_features, select_rows_by_class, replace_values,
    handle_missing_values, remove_duplicates, scale_data,
    encode_class, add_polynomial_features, remove_rows_by_ranges,
    replace_values_in_range
)
from src.utils import get_column_types, parse_row_ranges, parse_value_range
from components.descriptions import get_page_description, get_manipulation_method_description
from components.ui_helpers import show_info_box, section_header, display_metric_group


def page_data_manipulation():
    """Wyświetla stronę manipulacji danymi."""

    # Pobierz opis strony
    page_info = get_page_description("manipulation")

    # Nagłówek strony
    st.header(page_info["title"])
    st.markdown(page_info["description"])

    # Upewnij się, że data istnieje w session_state
    if 'data' not in st.session_state:
        st.error("Błąd: Dane nie zostały wczytane.")
        return

    # Upewnij się, że oryginalne dane są zachowane
    if 'original_data' not in st.session_state:
        st.session_state.original_data = st.session_state.data.copy()

    # Sekcja edycji danych w interfejsie
    st.subheader("📝 Edycja danych")
    with st.expander("✏️ Edytuj dane bezpośrednio", expanded=False):
        st.markdown("""
        **Instrukcja:** Poniżej możesz bezpośrednio edytować wartości w tabeli. 
        Zmiany zostaną automatycznie zastosowane po kliknięciu poza edytowaną komórkę.
        """)

        # Edytowalna tabela danych
        edited_data = st.data_editor(
            st.session_state.data,
            use_container_width=True,
            num_rows="dynamic",  # Pozwala dodawać/usuwać wiersze
            key="data_editor"
        )

        # Sprawdź czy dane zostały zmienione
        if not edited_data.equals(st.session_state.data):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Zastosuj zmiany", type="primary"):
                    st.session_state.data = edited_data.copy()
                    st.success("Zmiany zostały zastosowane!")
                    st.rerun()
            with col2:
                if st.button("↩️ Anuluj zmiany"):
                    st.info("Zmiany zostały anulowane.")
                    st.rerun()

    # Wyświetl podgląd aktualnych danych
    st.subheader("Podgląd aktualnych danych")
    st.dataframe(st.session_state.data.head(), use_container_width=True)

    # Menu operacji manipulacji danymi
    st.markdown("---")
    st.subheader("Wybierz operację")

    # Wyświetl opis operacji
    with st.expander("ℹ️ Dostępne operacje manipulacji danymi", expanded=True):
        st.markdown("""
        W tej sekcji możesz wykonać różne operacje manipulacji danymi:

        - **Wybierz cechy**: Wybierz, które kolumny mają zostać zachowane w zbiorze danych
        - **Wybierz wiersze według klasy**: Filtruj dane, zachowując tylko wybrane klasy win
        - **Usuń wiersze według numerów/zakresów**: Usuń konkretne wiersze podając numery lub zakresy (np. "1-5,8,10-12")
        - **Zastąp wartości**: Zastąp konkretne wartości w wybranej kolumnie innymi wartościami
        - **Zastąp wartości w zakresie**: Zastąp wszystkie wartości w określonym zakresie jedną wartością
        - **Obsłuż brakujące wartości**: Wypełnij brakujące wartości (NaN) różnymi metodami
        - **Usuń duplikaty**: Usuń powtarzające się wiersze z danych
        - **Skaluj dane**: Znormalizuj lub wystandaryzuj wartości w kolumnach
        - **Kodowanie binarne klasy**: Przekształć kolumnę Class w format one-hot encoding
        - **Dodaj cechy wielomianowe**: Dodaj nowe cechy będące wielomianami istniejących cech
        - **Resetuj dane**: Przywróć oryginalny zbiór danych
        """)

    # Wybór operacji
    operation = st.selectbox(
        "Wybierz operację do wykonania:",
        ["Wybierz cechy", "Wybierz wiersze według klasy", "Usuń wiersze według numerów/zakresów",
         "Zastąp wartości", "Zastąp wartości w zakresie", "Obsłuż brakujące wartości",
         "Usuń duplikaty", "Skaluj dane", "Kodowanie binarne klasy", "Dodaj cechy wielomianowe",
         "Resetuj dane"]
    )

    # Wykonaj wybraną operację
    if operation == "Wybierz cechy":
        method_info = get_manipulation_method_description("select_features")
        section_header(method_info["title"], "Wybierz, które kolumny zachować w zbiorze danych")

        with st.expander("ℹ️ Więcej informacji"):
            st.markdown(method_info["description"])

        # Pobierz listę wszystkich kolumn
        all_columns = st.session_state.data.columns.tolist()

        # Widget do wyboru kolumn
        selected_features = st.multiselect(
            "Wybierz kolumny do zachowania:",
            all_columns,
            default=all_columns,
            help="Wybierz kolumny, które chcesz zachować w danych. Pozostałe kolumny zostaną usunięte."
        )

        # Przycisk do zastosowania
        if st.button("Zastosuj", key="apply_select_features"):
            if selected_features:
                # Zastosuj wybór cech
                with st.spinner("Przetwarzanie danych..."):
                    st.session_state.data = select_features(st.session_state.data, selected_features)

                # Pokaż sukces
                st.success(f"Wybrano {len(selected_features)} kolumn. Pozostałe kolumny zostały usunięte.")
            else:
                st.error("Wybierz co najmniej jedną kolumnę.")

    elif operation == "Wybierz wiersze według klasy":
        method_info = get_manipulation_method_description("select_rows")
        section_header(method_info["title"], "Filtruj dane według wartości kolumny Class")

        with st.expander("ℹ️ Więcej informacji"):
            st.markdown(method_info["description"])

        # Sprawdź czy kolumna 'Class' istnieje
        if 'Class' in st.session_state.data.columns:
            # Pobierz unikalne wartości klasy
            available_classes = sorted(st.session_state.data['Class'].unique())

            # Widget do wyboru klas
            selected_classes = st.multiselect(
                "Wybierz klasy do zachowania:",
                available_classes,
                default=available_classes,
                help="Wybierz klasy win, które chcesz zachować w danych. Wiersze z pozostałymi klasami zostaną usunięte."
            )

            # Przycisk do zastosowania
            if st.button("Zastosuj", key="apply_select_rows"):
                if selected_classes:
                    # Zastosuj filtrowanie według klasy
                    with st.spinner("Przetwarzanie danych..."):
                        orig_len = len(st.session_state.data)
                        st.session_state.data = select_rows_by_class(st.session_state.data, selected_classes)
                        new_len = len(st.session_state.data)

                    # Pokaż sukces
                    st.success(f"Wybrano wiersze dla klas: {selected_classes}. Usunięto {orig_len - new_len} wierszy.")
                else:
                    st.error("Wybierz co najmniej jedną klasę.")
        else:
            st.error("Kolumna 'Class' nie istnieje w danych.")

    elif operation == "Usuń wiersze według numerów/zakresów":
        section_header("Usuń wiersze według numerów/zakresów", "Usuń konkretne wiersze podając ich numery lub zakresy")

        with st.expander("ℹ️ Instrukcja używania zakresów", expanded=True):
            st.markdown("""
            **Jak podawać zakresy wierszy:**
            
            - **Pojedynczy wiersz**: `5` (usuwa wiersz numer 5)
            - **Zakres wierszy**: `10-15` (usuwa wiersze od 10 do 15 włącznie)
            - **Kombinacja**: `1-3,7,10-12,20` (usuwa wiersze 1-3, 7, 10-12 i 20)
            
            **Przykłady poprawnego formatu:**
            - `1,3,5` - usuwa wiersze 1, 3 i 5
            - `10-20` - usuwa wiersze od 10 do 20
            - `1-5,8,15-20` - usuwa wiersze 1-5, wiersz 8 i wiersze 15-20
            
            **Uwaga:** Numeracja wierszy zaczyna się od 0!
            """)

        # Informacja o aktualnej liczbie wierszy
        total_rows = len(st.session_state.data)
        st.info(f"Aktualnie w zbiorze danych jest {total_rows} wierszy (indeksy 0-{total_rows-1})")

        # Pole do wprowadzenia zakresów
        row_ranges = st.text_input(
            "Podaj numery/zakresy wierszy do usunięcia:",
            placeholder="np. 1-5,8,10-12",
            help="Wprowadź numery wierszy lub zakresy oddzielone przecinkami"
        )

        # Przycisk do zastosowania
        if st.button("Usuń wiersze", key="apply_remove_rows"):
            if row_ranges.strip():
                try:
                    # Parsuj zakresy
                    rows_to_remove = parse_row_ranges(row_ranges, total_rows)

                    if rows_to_remove:
                        # Zastosuj usunięcie wierszy
                        with st.spinner("Usuwanie wierszy..."):
                            st.session_state.data = remove_rows_by_ranges(st.session_state.data, rows_to_remove)

                        # Pokaż sukces
                        st.success(f"Usunięto {len(rows_to_remove)} wierszy. Pozostało {len(st.session_state.data)} wierszy.")
                    else:
                        st.warning("Nie znaleziono poprawnych wierszy do usunięcia.")

                except ValueError as e:
                    st.error(f"Błąd w formacie zakresów: {str(e)}")
            else:
                st.error("Wprowadź zakresy wierszy do usunięcia.")

    elif operation == "Zastąp wartości":
        method_info = get_manipulation_method_description("replace_values")
        section_header(method_info["title"], "Zastąp określone wartości w wybranej kolumnie")

        with st.expander("ℹ️ Więcej informacji"):
            st.markdown(method_info["description"])

        # Widget do wyboru kolumny
        column = st.selectbox(
            "Wybierz kolumnę:",
            st.session_state.data.columns.tolist(),
            help="Wybierz kolumnę, w której chcesz zastąpić wartości"
        )

        if column:
            # Sprawdź typ kolumny
            col_type = st.session_state.data[column].dtype

            # Jeśli kolumna jest numeryczna
            if np.issubdtype(col_type, np.number):
                # Widget do wprowadzenia starej wartości
                old_value = st.number_input(
                    "Stara wartość:",
                    value=0.0,
                    step=0.1,
                    help="Wprowadź wartość, którą chcesz zastąpić"
                )

                # Widget do wyboru typu nowej wartości
                new_value_type = st.selectbox(
                    "Typ nowej wartości:",
                    ["Liczba", "NaN"],
                    help="Wybierz, czy chcesz zastąpić starą wartość liczbą czy wartością pustą (NaN)"
                )

                # Widget do wprowadzenia nowej wartości (jeśli wybrano liczbę)
                if new_value_type == "Liczba":
                    new_value = st.number_input(
                        "Nowa wartość:",
                        value=0.0,
                        step=0.1,
                        help="Wprowadź nową wartość"
                    )
                else:
                    new_value = None

                # Przycisk do zastosowania
                if st.button("Zastosuj", key="apply_replace_values"):
                    # Zastosuj zastąpienie wartości
                    with st.spinner("Przetwarzanie danych..."):
                        st.session_state.data = replace_values(
                            st.session_state.data, column, old_value, new_value
                        )

                    # Pokaż sukces
                    if new_value_type == "NaN":
                        st.success(f"Zastąpiono wartości {old_value} na NaN w kolumnie {column}.")
                    else:
                        st.success(f"Zastąpiono wartości {old_value} na {new_value} w kolumnie {column}.")

            # Jeśli kolumna nie jest numeryczna
            else:
                # Widgety do wprowadzenia wartości tekstowych
                old_value = st.text_input(
                    "Stara wartość:",
                    help="Wprowadź wartość tekstową, którą chcesz zastąpić"
                )
                new_value = st.text_input(
                    "Nowa wartość:",
                    help="Wprowadź nową wartość tekstową"
                )

                # Przycisk do zastosowania
                if st.button("Zastosuj", key="apply_replace_text"):
                    # Zastosuj zastąpienie wartości
                    with st.spinner("Przetwarzanie danych..."):
                        st.session_state.data = replace_values(
                            st.session_state.data, column, old_value, new_value
                        )

                    # Pokaż sukces
                    st.success(f"Zastąpiono wartości '{old_value}' na '{new_value}' w kolumnie {column}.")

    elif operation == "Zastąp wartości w zakresie":
        section_header("Zastąp wartości w zakresie", "Zastąp wszystkie wartości w określonym zakresie jedną wartością")

        with st.expander("ℹ️ Instrukcja używania zakresów wartości", expanded=True):
            st.markdown("""
            **Jak podawać zakresy wartości:**
            
            - **Format**: `min-max` gdzie min i max to liczby rzeczywiste
            - **Przykłady**: 
              - `0.5-0.7` - zastąpi wszystkie wartości między 0.5 a 0.7 (włącznie)
              - `10-20` - zastąpi wszystkie wartości między 10 a 20
              - `0.1-0.3` - zastąpi wszystkie wartości między 0.1 a 0.3
            
            **Uwaga:** Operacja dotyczy tylko kolumn numerycznych!
            """)

        # Pobierz kolumny numeryczne
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        if not numeric_cols:
            st.error("Brak kolumn numerycznych w danych.")
        else:
            # Widget do wyboru kolumny
            column = st.selectbox(
                "Wybierz kolumnę numeryczną:",
                numeric_cols,
                help="Wybierz kolumnę, w której chcesz zastąpić wartości w zakresie"
            )

            if column:
                # Pokaż statystyki kolumny
                col_min = st.session_state.data[column].min()
                col_max = st.session_state.data[column].max()
                st.info(f"Kolumna '{column}': min = {col_min:.3f}, max = {col_max:.3f}")

                # Pole do wprowadzenia zakresu
                value_range = st.text_input(
                    "Zakres wartości do zastąpienia:",
                    placeholder="np. 0.5-0.7",
                    help="Wprowadź zakres w formacie 'min-max'"
                )

                # Nowa wartość
                new_value = st.number_input(
                    "Nowa wartość:",
                    value=0.0,
                    step=0.1,
                    help="Wartość, którą zostaną zastąpione wszystkie wartości w zakresie"
                )

                # Przycisk do zastosowania
                if st.button("Zastąp wartości w zakresie", key="apply_replace_range"):
                    if value_range.strip():
                        try:
                            # Parsuj zakres
                            min_val, max_val = parse_value_range(value_range)

                            # Sprawdź czy zakres jest w granicach danych
                            if min_val > col_max or max_val < col_min:
                                st.warning(f"Zakres {min_val}-{max_val} nie pokrywa się z wartościami w kolumnie.")
                            else:
                                # Zastosuj zastąpienie wartości w zakresie
                                with st.spinner("Zastępowanie wartości..."):
                                    affected_rows = st.session_state.data[
                                        (st.session_state.data[column] >= min_val) &
                                        (st.session_state.data[column] <= max_val)
                                    ].shape[0]

                                    st.session_state.data = replace_values_in_range(
                                        st.session_state.data, column, min_val, max_val, new_value
                                    )

                                # Pokaż sukces
                                st.success(f"Zastąpiono {affected_rows} wartości z zakresu {min_val}-{max_val} na {new_value} w kolumnie '{column}'.")

                        except ValueError as e:
                            st.error(f"Błąd w formacie zakresu: {str(e)}")
                    else:
                        st.error("Wprowadź zakres wartości.")

    elif operation == "Obsłuż brakujące wartości":
        method_info = get_manipulation_method_description("missing_values")
        section_header(method_info["title"], "Wypełnij brakujące wartości (NaN) w danych")

        with st.expander("ℹ️ Więcej informacji"):
            st.markdown(method_info["description"])

        # Widget do wyboru strategii
        strategy = st.selectbox(
            "Wybierz strategię wypełniania brakujących wartości:",
            ["mean", "median", "most_frequent", "constant"],
            index=0,
            help="""
            - Mean: zastępuje braki średnią wartością kolumny
            - Median: zastępuje braki medianą wartości kolumny
            - Most frequent: zastępuje braki najczęściej występującą wartością
            - Constant: zastępuje braki stałą wartością
            """
        )

        # Jeśli wybrano strategię constant, pokaż pole do wprowadzenia wartości
        if strategy == "constant":
            fill_value = st.number_input(
                "Wartość do wypełnienia:",
                value=0.0,
                help="Wprowadź wartość, którą zostaną wypełnione wszystkie braki"
            )

        # Sprawdź, czy są jakieś brakujące wartości
        missing_count = st.session_state.data.isnull().sum().sum()

        if missing_count > 0:
            st.info(f"Znaleziono {missing_count} brakujących wartości w danych.")
        else:
            st.info("Nie znaleziono brakujących wartości w danych.")

        # Przycisk do zastosowania
        if st.button("Zastosuj", key="apply_missing_values"):
            # Zastosuj obsługę brakujących wartości
            with st.spinner("Przetwarzanie danych..."):
                params = {"strategy": strategy}
                if strategy == "constant":
                    params["fill_value"] = fill_value

                st.session_state.data = handle_missing_values(st.session_state.data, **params)

            # Pokaż sukces
            if missing_count > 0:
                st.success(f"Uzupełniono {missing_count} brakujących wartości strategią: {strategy}.")
            else:
                st.success("Nie było brakujących wartości do wypełnienia.")

    elif operation == "Usuń duplikaty":
        method_info = get_manipulation_method_description("duplicates")
        section_header(method_info["title"], "Usuń powtarzające się wiersze z danych")

        with st.expander("ℹ️ Więcej informacji"):
            st.markdown(method_info["description"])

        # Sprawdź, czy są jakieś duplikaty
        duplicate_count = st.session_state.data.duplicated().sum()

        if duplicate_count > 0:
            st.info(f"Znaleziono {duplicate_count} duplikatów w danych.")
        else:
            st.info("Nie znaleziono duplikatów w danych.")

        # Przycisk do zastosowania
        if st.button("Zastosuj", key="apply_remove_duplicates"):
            # Zastosuj usunięcie duplikatów
            with st.spinner("Przetwarzanie danych..."):
                original_len = len(st.session_state.data)
                st.session_state.data = remove_duplicates(st.session_state.data)
                new_len = len(st.session_state.data)

            # Pokaż sukces
            if original_len > new_len:
                st.success(f"Usunięto {original_len - new_len} duplikatów.")
            else:
                st.info("Nie znaleziono duplikatów do usunięcia.")

    elif operation == "Skaluj dane":
        method_info = get_manipulation_method_description("scaling")
        section_header(method_info["title"], "Skaluj wartości numeryczne do określonego zakresu")

        with st.expander("ℹ️ Więcej informacji"):
            st.markdown(method_info["description"])

        # Widget do wyboru metody skalowania
        method = st.selectbox(
            "Wybierz metodę skalowania:",
            ["standard", "minmax"],
            index=0,
            help="""
            - Standard: standaryzacja (z-score), (x - mean) / std, średnia=0, odch.std=1
            - MinMax: skalowanie do zakresu [0,1], (x - min) / (max - min)
            """
        )

        # Pobierz kolumny numeryczne
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        # Usuń kolumnę Class z listy, jeśli istnieje
        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')

        # Widget do wyboru kolumn do skalowania
        columns_to_scale = st.multiselect(
            "Wybierz kolumny do skalowania:",
            numeric_cols,
            default=numeric_cols,
            help="Wybierz kolumny numeryczne, które chcesz przeskalować"
        )

        # Przycisk do zastosowania
        if st.button("Zastosuj", key="apply_scaling"):
            if columns_to_scale:
                # Zastosuj skalowanie danych
                with st.spinner("Przetwarzanie danych..."):
                    st.session_state.data = scale_data(
                        st.session_state.data, method, columns_to_scale
                    )

                # Pokaż sukces
                st.success(f"Przeskalowano dane metodą: {method} dla kolumn: {', '.join(columns_to_scale)}")
            else:
                st.error("Wybierz co najmniej jedną kolumnę do skalowania.")

    elif operation == "Kodowanie binarne klasy":
        method_info = get_manipulation_method_description("encoding")
        section_header(method_info["title"], "Przekształć kolumnę Class w format one-hot encoding")

        with st.expander("ℹ️ Więcej informacji"):
            st.markdown(method_info["description"])

        # Sprawdź czy kolumna 'Class' istnieje
        if 'Class' in st.session_state.data.columns:
            # Wizualizacja przykładu kodowania
            with st.expander("🔄 Przykład kodowania one-hot"):
                st.markdown("""
                **Przed kodowaniem:**
                | Class |
                |-------|
                | 1     |
                | 2     |
                | 3     |

                **Po kodowaniu:**
                | Class_1 | Class_2 | Class_3 |
                |---------|---------|---------|
                | 1       | 0       | 0       |
                | 0       | 1       | 0       |
                | 0       | 0       | 1       |
                """)

            # Przycisk do zastosowania
            if st.button("Zastosuj", key="apply_encoding"):
                # Zastosuj kodowanie klasy
                with st.spinner("Przetwarzanie danych..."):
                    st.session_state.data = encode_class(st.session_state.data)

                # Pokaż sukces
                st.success("Zastosowano kodowanie one-hot dla kolumny 'Class'.")

                # Pokaż nowe kolumny
                st.info("Utworzono nowe kolumny: " + ", ".join(
                    [col for col in st.session_state.data.columns if col.startswith('Class_')]))
        else:
            st.error("Kolumna 'Class' nie istnieje w danych.")

    elif operation == "Dodaj cechy wielomianowe":
        method_info = get_manipulation_method_description("polynomial")
        section_header(method_info["title"], "Dodaj nowe cechy będące wielomianami istniejących cech")

        with st.expander("ℹ️ Więcej informacji"):
            st.markdown(method_info["description"])

            # Dodaj przykład tworzenia cech wielomianowych
            st.markdown("""
            **Przykład:**

            Dla cechy "Alcohol" o wartości 12.5, przy stopniu wielomianu 2, zostanie dodana nowa cecha:
            - "Alcohol^2" = 12.5² = 156.25

            A przy stopniu wielomianu 3, zostaną dodane dwie nowe cechy:
            - "Alcohol^2" = 12.5² = 156.25
            - "Alcohol^3" = 12.5³ = 1953.125
            """)

        # Pobierz kolumny numeryczne
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        # Usuń kolumnę Class z listy, jeśli istnieje
        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')

        # Widget do wyboru kolumn do transformacji
        columns_to_transform = st.multiselect(
            "Wybierz kolumny do transformacji:",
            numeric_cols,
            help="Wybierz kolumny numeryczne, dla których chcesz dodać cechy wielomianowe"
        )

        # Widget do wyboru stopnia wielomianu
        degree = st.slider(
            "Stopień wielomianu:",
            min_value=2,
            max_value=5,
            value=2,
            help="Wybierz maksymalny stopień wielomianu (np. 2 oznacza dodanie X²)"
        )

        # Przycisk do zastosowania
        if st.button("Zastosuj", key="apply_polynomial"):
            if columns_to_transform:
                # Zastosuj dodanie cech wielomianowych
                with st.spinner("Przetwarzanie danych..."):
                    orig_cols = st.session_state.data.shape[1]
                    st.session_state.data = add_polynomial_features(
                        st.session_state.data, columns_to_transform, degree
                    )
                    new_cols = st.session_state.data.shape[1]

                # Pokaż sukces
                st.success(f"Dodano {new_cols - orig_cols} cech wielomianowych stopnia {degree}.")

                # Pokaż nowe kolumny
                new_col_names = [col for col in st.session_state.data.columns if '^' in col]
                if new_col_names:
                    st.info("Utworzono nowe kolumny: " + ", ".join(new_col_names[:10]))
                    if len(new_col_names) > 10:
                        st.info(f"...i {len(new_col_names) - 10} więcej.")
            else:
                st.error("Wybierz co najmniej jedną kolumnę do transformacji.")

    elif operation == "Resetuj dane":
        section_header("Resetuj dane", "Przywróć oryginalny zbiór danych")

        st.markdown("""
        Ta operacja przywraca oryginalny zbiór danych, anulując wszystkie wcześniejsze operacje manipulacji danymi.
        Wszystkie zmiany wprowadzone w danych zostaną utracone.
        """)

        # Przycisk do resetowania danych
        if st.button("Resetuj dane do stanu początkowego", key="reset_data"):
            # Resetuj dane
            with st.spinner("Przywracanie oryginalnych danych..."):
                st.session_state.data = st.session_state.original_data.copy()

            # Pokaż sukces
            st.success("Dane zostały zresetowane do stanu początkowego.")

    # Wyświetl podgląd zmodyfikowanych danych
    st.markdown("---")
    st.subheader("Podgląd zmodyfikowanych danych")
    st.dataframe(st.session_state.data.head(), use_container_width=True)

    # Informacje o aktualnych danych
    st.subheader("Informacje o aktualnych danych")
    info = get_dataset_info(st.session_state.data)

    # Wyświetl metryki w dwóch kolumnach
    metrics = {
        "Liczba wierszy": info["liczba_wierszy"],
        "Liczba kolumn": info["liczba_kolumn"],
        "Brakujące wartości": info["brakujące_wartości"],
        "Duplikaty": info["duplikaty"]
    }
    display_metric_group(metrics)

    # Podsumowanie
    st.markdown("---")
    st.subheader("Podsumowanie zmian")

    # Oblicz zmiany w stosunku do oryginalnych danych
    orig_rows = st.session_state.original_data.shape[0]
    orig_cols = st.session_state.original_data.shape[1]
    curr_rows = st.session_state.data.shape[0]
    curr_cols = st.session_state.data.shape[1]

    change_rows = curr_rows - orig_rows
    change_cols = curr_cols - orig_cols

    # Wyświetl opis zmian
    if change_rows == 0 and change_cols == 0:
        st.info("Nie wprowadzono zmian w strukturze danych.")
    else:
        changes = []

        if change_rows > 0:
            changes.append(f"Dodano {change_rows} wierszy.")
        elif change_rows < 0:
            changes.append(f"Usunięto {abs(change_rows)} wierszy.")

        if change_cols > 0:
            changes.append(f"Dodano {change_cols} kolumn.")
        elif change_cols < 0:
            changes.append(f"Usunięto {abs(change_cols)} kolumn.")

        st.info(" ".join(changes))

    # Porównanie kolumn
    if set(st.session_state.original_data.columns) != set(st.session_state.data.columns):
        # Nowe kolumny
        new_cols = [col for col in st.session_state.data.columns if col not in st.session_state.original_data.columns]
        if new_cols:
            st.success(f"Nowe kolumny: {', '.join(new_cols)}")

        # Usunięte kolumny
        removed_cols = [col for col in st.session_state.original_data.columns if
                        col not in st.session_state.data.columns]
        if removed_cols:
            st.warning(f"Usunięte kolumny: {', '.join(removed_cols)}")