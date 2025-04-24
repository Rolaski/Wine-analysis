"""
Moduł odpowiedzialny za stronę analizy statystycznej w aplikacji Wine Dataset Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modułów własnych
from src.statistical_analyzer import (
    calculate_basic_stats, calculate_quartiles, calculate_correlation_matrix,
    find_highly_correlated_features, calculate_class_stats, test_normality,
    calculate_outliers
)
from src.data_visualizer import create_histogram, create_boxplot
from src.utils import get_column_types
from components.descriptions import get_page_description, get_stat_method_description
from components.ui_helpers import show_info_box, section_header, show_feature_description


def page_statistical_analysis():
    """Wyświetla stronę analizy statystycznej."""

    # Pobierz opis strony
    page_info = get_page_description("statistical")

    # Nagłówek strony
    st.header(page_info["title"])
    st.markdown(page_info["description"])

    # Wybór kolumn do analizy
    st.subheader("Wybierz kolumny do analizy")

    # Pobierz typy kolumn
    column_types = get_column_types(st.session_state.data)

    # Wybierz wszystkie kolumny numeryczne z wyjątkiem Class
    all_columns = column_types.get('numeric', [])
    if 'Class' in all_columns:
        all_columns.remove('Class')

    # Multiselect do wyboru kolumn
    selected_columns = st.multiselect(
        "Wybierz kolumny numeryczne do analizy:",
        all_columns,
        default=all_columns[:5] if len(all_columns) > 5 else all_columns,
        help="Wybierz kolumny, które chcesz analizować. Możesz wybrać wiele kolumn."
    )

    if not selected_columns:
        st.warning("Wybierz co najmniej jedną kolumnę do analizy.")
        return

    # Dodaj odstęp
    st.markdown("---")

    # Podstawowe statystyki z opisem
    method_info = get_stat_method_description("podstawowe_statystyki")
    section_header(method_info["title"], "Podstawowe miary statystyczne dla wybranych kolumn")

    with st.expander("ℹ️ Co oznaczają te statystyki?", expanded=True):
        st.markdown(method_info["description"])

    # Oblicz podstawowe statystyki
    basic_stats = calculate_basic_stats(st.session_state.data, selected_columns)

    # Wyświetl statystyki
    st.dataframe(basic_stats.round(3), use_container_width=True)

    # Kwartyle i percentyle z opisem
    method_info = get_stat_method_description("kwartyle")
    section_header(method_info["title"], "Miary pozycyjne dla wybranych kolumn")

    with st.expander("ℹ️ Co to są kwartyle i percentyle?", expanded=True):
        st.markdown(method_info["description"])

    # Oblicz kwartyle
    quartiles = calculate_quartiles(st.session_state.data, selected_columns)

    # Wyświetl kwartyle
    st.dataframe(quartiles.round(3), use_container_width=True)

    # Test normalności z opisem
    method_info = get_stat_method_description("test_normalnosci")
    section_header(method_info["title"], "Sprawdzenie czy dane pochodzą z rozkładu normalnego")

    with st.expander("ℹ️ Co to jest test normalności i dlaczego jest ważny?", expanded=True):
        st.markdown(method_info["description"])

    # Oblicz test normalności
    normality = test_normality(st.session_state.data, selected_columns)

    # Wyświetl wyniki testu
    st.dataframe(normality.round(4), use_container_width=True)

    # Wykresy dla wybranych kolumn
    st.markdown("---")
    st.subheader("Wizualizacja danych dla wybranych kolumn")
    st.markdown("Poniższe wykresy pomagają zrozumieć rozkład i charakterystykę wybranych zmiennych.")

    # Wybór kolumny do wizualizacji
    column_to_plot = st.selectbox(
        "Wybierz kolumnę do wizualizacji:",
        selected_columns,
        help="Wybierz kolumnę, dla której chcesz zobaczyć wykresy."
    )

    # Pokaż opis wybranej cechy
    show_feature_description(column_to_plot)

    # Ustawienia histogramu
    st.subheader(f"Histogram dla {column_to_plot}")

    with st.expander("ℹ️ Jak interpretować histogram?"):
        st.markdown("""
        Histogram pokazuje rozkład wartości w wybranej kolumnie. Oś X przedstawia przedziały wartości, 
        a oś Y liczbę obserwacji, które mieszczą się w danym przedziale.

        - Gdy zaznaczysz opcję "Grupuj według klasy", zobaczysz oddzielne histogramy dla każdej klasy wina.
        - Liczba przedziałów określa, jak szczegółowy będzie histogram.

        Histogramy pomagają zrozumieć:
        - Kształt rozkładu (symetryczny, skośny, wielomodalny)
        - Zakres wartości
        - Potencjalne wartości odstające
        - Różnice między klasami
        """)

    # Opcje histogramu
    hist_col1, hist_col2 = st.columns(2)

    with hist_col1:
        by_class = st.checkbox("Grupuj według klasy", value=True,
                               help="Pokaż oddzielne histogramy dla każdej klasy wina")

    with hist_col2:
        bins = st.slider("Liczba przedziałów:", 5, 50, 20,
                         help="Większa liczba przedziałów daje bardziej szczegółowy histogram")

    # Rysuj histogram
    hist_fig = create_histogram(st.session_state.data, column_to_plot, bins, by_class)
    st.pyplot(hist_fig)

    # Wykres pudełkowy
    st.subheader(f"Wykres pudełkowy dla {column_to_plot}")

    with st.expander("ℹ️ Jak interpretować wykres pudełkowy?"):
        st.markdown("""
        Wykres pudełkowy (boxplot) pokazuje rozkład wartości i potencjalne wartości odstające:

        - **Środkowa linia**: mediana (wartość środkowa)
        - **Dolna krawędź pudełka**: pierwszy kwartyl (Q1, 25%)
        - **Górna krawędź pudełka**: trzeci kwartyl (Q3, 75%)
        - **Wąsy**: rozciągają się do min/max wartości (z wyłączeniem wartości odstających)
        - **Punkty poza wąsami**: potencjalne wartości odstające

        Wykresy pudełkowe są szczególnie użyteczne do:
        - Porównywania rozkładów między grupami (np. klasami win)
        - Identyfikowania wartości odstających
        - Oceny rozproszenia i symetrii danych
        """)

    # Rysuj wykres pudełkowy
    box_fig = create_boxplot(st.session_state.data, column_to_plot, by_class=True)
    st.pyplot(box_fig)

    # Analiza korelacji
    st.markdown("---")
    method_info = get_stat_method_description("korelacje")
    section_header(method_info["title"], "Badanie zależności między zmiennymi")

    with st.expander("ℹ️ Co to jest korelacja i jak ją interpretować?", expanded=True):
        st.markdown(method_info["description"])

    # Wybór metody korelacji
    corr_method = st.selectbox(
        "Wybierz metodę korelacji:",
        ["pearson", "spearman", "kendall"],
        index=0,
        help="""
        - Pearson: mierzy liniową zależność, wymaga danych normalnych
        - Spearman: mierzy monotoniczną zależność, nie wymaga normalności
        - Kendall: podobna do Spearman, ale bardziej odporna na wartości odstające
        """
    )

    # Próg korelacji do wyświetlenia par
    corr_threshold = st.slider(
        "Próg korelacji dla par cech:",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Pokaż tylko pary cech o korelacji (bezwzględnej) większej niż ten próg"
    )

    # Macierz korelacji
    st.subheader(f"Macierz korelacji ({corr_method})")

    # Oblicz macierz korelacji
    corr_matrix = calculate_correlation_matrix(
        st.session_state.data,
        method=corr_method,
        columns=selected_columns
    )

    # Wyświetl macierz korelacji
    st.dataframe(corr_matrix.round(2), use_container_width=True)

    # Pary cech o wysokiej korelacji
    st.subheader(f"Pary cech o wysokiej korelacji (>{corr_threshold})")

    # Znajdź pary cech o wysokiej korelacji
    high_corr = find_highly_correlated_features(
        st.session_state.data,
        threshold=corr_threshold,
        method=corr_method
    )

    # Wyświetl pary cech o wysokiej korelacji
    if high_corr is not None and not high_corr.empty:
        st.dataframe(high_corr.round(3), use_container_width=True)

        # Interpretacja wyników
        st.markdown("""
        **Interpretacja:**
        - Cechy o wysokiej dodatniej korelacji (bliskiej 1) zmieniają się razem w tym samym kierunku
        - Cechy o wysokiej ujemnej korelacji (bliskiej -1) zmieniają się w przeciwnych kierunkach
        - Silnie skorelowane cechy mogą zawierać redundantne informacje
        """)
    else:
        st.info(f"Brak par cech o korelacji (bezwzględnej) większej niż {corr_threshold}.")

    # Statystyki według klas
    st.markdown("---")
    st.subheader("Statystyki według klas")

    with st.expander("ℹ️ O statystykach według klas"):
        st.markdown("""
        Ta sekcja pokazuje podstawowe statystyki dla każdej klasy wina oddzielnie.
        Pozwala to na porównanie różnic w charakterystykach chemicznych między klasami win.

        Dla każdej klasy pokazane są te same statystyki co w sekcji "Podstawowe statystyki",
        ale obliczone tylko dla win z wybranej klasy.
        """)

    # Oblicz statystyki według klas
    class_stats = calculate_class_stats(st.session_state.data, selected_columns)

    # Wybór klasy do wyświetlenia
    class_to_show = st.selectbox(
        "Wybierz klasę:",
        sorted(class_stats.keys()),
        help="Wybierz klasę wina, dla której chcesz zobaczyć statystyki"
    )

    # Wyświetl statystyki dla wybranej klasy
    if class_to_show in class_stats:
        st.dataframe(class_stats[class_to_show].round(3), use_container_width=True)

    # Analiza wartości odstających
    st.markdown("---")
    method_info = get_stat_method_description("wartosci_odstajace")
    section_header(method_info["title"], "Identyfikacja nietypowych obserwacji")

    with st.expander("ℹ️ Co to są wartości odstające i jak je identyfikować?", expanded=True):
        st.markdown(method_info["description"])

    # Wybór metody wykrywania wartości odstających
    outlier_method = st.selectbox(
        "Wybierz metodę wykrywania wartości odstających:",
        ["iqr", "zscore"],
        index=0,
        help="""
        - IQR: wartości poza zakresem [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        - Z-score: wartości oddalone od średniej o więcej niż 3 odchylenia standardowe
        """
    )

    # Wybór kolumny do analizy wartości odstających
    outlier_col = st.selectbox(
        "Wybierz kolumnę do analizy wartości odstających:",
        selected_columns,
        help="Wybierz kolumnę, dla której chcesz znaleźć wartości odstające"
    )

    # Oblicz wartości odstające
    outliers = calculate_outliers(
        st.session_state.data,
        [outlier_col],
        method=outlier_method
    )

    # Wyświetl wartości odstające
    if outlier_col in outliers and not outliers[outlier_col].empty:
        st.write(f"Znaleziono {len(outliers[outlier_col])} wartości odstających dla {outlier_col}:")
        st.dataframe(outliers[outlier_col])

        # Wizualizacja wartości odstających
        st.subheader("Wizualizacja wartości odstających")

        # Wybór typu wykresu
        viz_type = st.radio(
            "Wybierz typ wizualizacji:",
            ["Histogram", "Wykres pudełkowy"],
            horizontal=True
        )

        if viz_type == "Histogram":
            # Rysuj histogram z zaznaczonymi wartościami odstającymi
            fig, ax = plt.subplots(figsize=(10, 6))

            # Wszystkie dane
            sns.histplot(st.session_state.data[outlier_col], bins=20, kde=True, ax=ax, color='blue', alpha=0.5)

            # Wartości odstające
            outlier_values = outliers[outlier_col][outlier_col].values
            if len(outlier_values) > 0:
                sns.rugplot(outlier_values, ax=ax, color='red', height=0.1, label='Wartości odstające')

            plt.title(f'Histogram {outlier_col} z zaznaczonymi wartościami odstającymi')
            plt.legend()

            st.pyplot(fig)
        else:
            # Rysuj wykres pudełkowy
            fig = create_boxplot(st.session_state.data, outlier_col, by_class=True)
            st.pyplot(fig)
    else:
        st.info(f"Nie znaleziono wartości odstających dla {outlier_col} metodą {outlier_method}.")

    # Podsumowanie
    st.markdown("---")
    st.subheader("Podsumowanie analizy statystycznej")
    show_info_box("Główne wnioski", """
    Na podstawie przeprowadzonej analizy statystycznej można wyciągnąć następujące wnioski:

    1. **Rozkłady danych**: Większość cech nie ma rozkładu normalnego, co może wpływać na wybór metod statystycznych.
    2. **Korelacje**: Istnieją silne korelacje między niektórymi cechami, co wskazuje na potencjalną redundancję informacji.
    3. **Różnice między klasami**: Widoczne są różnice w charakterystykach chemicznych między trzema klasami win.
    4. **Wartości odstające**: Zidentyfikowano kilka wartości odstających, które mogą wymagać specjalnego traktowania.

    Informacje te są cenne przy dalszej eksploracji danych i budowie modeli predykcyjnych.
    """)