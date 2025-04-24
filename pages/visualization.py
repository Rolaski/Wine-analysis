"""
Moduł odpowiedzialny za stronę wizualizacji danych w aplikacji Wine Dataset Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modułów własnych
from src.data_visualizer import (
    create_histogram, create_boxplot, create_scatter_plot,
    create_correlation_heatmap, create_pairplot, create_3d_scatter,
    create_parallel_coordinates, create_class_distribution
)
from src.utils import get_column_types
from components.descriptions import get_page_description, get_visualization_method_description
from components.ui_helpers import show_info_box, section_header, show_feature_description


def page_visualization():
    """Wyświetla stronę wizualizacji danych."""

    # Pobierz opis strony
    page_info = get_page_description("visualization")

    # Nagłówek strony
    st.header(page_info["title"])
    st.markdown(page_info["description"])

    # Wprowadzenie do wizualizacji danych
    with st.expander("ℹ️ Dlaczego wizualizacja danych jest ważna?", expanded=True):
        st.markdown("""
        Wizualizacja danych to potężne narzędzie analityczne, które pozwala:

        - **Zobaczyć wzorce i trendy** które mogą być niewidoczne w surowych danych
        - **Zidentyfikować wartości odstające** i anomalie w danych
        - **Odkryć korelacje i zależności** między różnymi cechami
        - **Lepiej zrozumieć strukturę danych** i rozkłady zmiennych
        - **Efektywnie komunikować wyniki** analiz innym osobom

        W tej sekcji możesz tworzyć różnorodne wizualizacje dla zbioru danych Wine Dataset, 
        aby lepiej zrozumieć charakterystyki win i zależności między ich cechami chemicznymi.
        """)

    # Wybór typu wizualizacji
    st.markdown("---")
    st.subheader("Wybierz typ wizualizacji")

    visualization_type = st.selectbox(
        "Typ wizualizacji:",
        ["Histogram", "Wykres pudełkowy", "Wykres rozproszenia", "Wykres rozproszenia 3D",
         "Macierz korelacji", "Wykres par cech", "Współrzędne równoległe",
         "Rozkład klas"],
        help="Wybierz rodzaj wykresu do utworzenia"
    )

    # Pobierz typy kolumn
    column_types = get_column_types(st.session_state.data)
    numeric_cols = column_types.get('numeric', [])

    # Histogram
    if visualization_type == "Histogram":
        method_info = get_visualization_method_description("histogram")
        section_header(method_info["title"], "Wizualizacja rozkładu wartości cechy")

        with st.expander("ℹ️ Jak interpretować histogram?"):
            st.markdown(method_info["description"])

        # Wybór kolumny
        column = st.selectbox(
            "Wybierz kolumnę:",
            numeric_cols,
            help="Wybierz kolumnę, dla której chcesz utworzyć histogram"
        )

        if column:
            # Pokaż opis wybranej cechy
            show_feature_description(column)

            # Opcje histogramu
            col1, col2 = st.columns(2)

            with col1:
                bins = st.slider(
                    "Liczba przedziałów:",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Większa liczba przedziałów daje bardziej szczegółowy histogram"
                )

            with col2:
                by_class = st.checkbox(
                    "Grupuj według klasy",
                    value=True,
                    help="Pokaż oddzielne histogramy dla każdej klasy wina"
                )

            # Tworzenie histogramu
            fig = create_histogram(st.session_state.data, column, bins, by_class)
            st.pyplot(fig)

            # Wskazówki interpretacji
            with st.expander("💡 Wskazówki interpretacji"):
                st.markdown(f"""
                **Co można odczytać z tego histogramu:**

                - **Zakres wartości**: {column} przyjmuje wartości od około {st.session_state.data[column].min():.2f} do {st.session_state.data[column].max():.2f}
                - **Rozproszenie**: Zobaczysz, czy wartości są skupione wokół jednej wartości, czy rozproszone
                - **Skośność**: Jeśli wykres jest przesunięty w lewo lub prawo, wskazuje to na skośność rozkładu
                - **Wielomodalność**: Kilka "szczytów" może sugerować istnienie podgrup w danych

                **Porównanie klas:**
                - Zwróć uwagę na różnice w rozkładach między klasami
                - Jeśli rozkłady wyraźnie się różnią, może to wskazywać, że ta cecha jest dobrym predyktorem klasy wina
                """)

    # Wykres pudełkowy
    elif visualization_type == "Wykres pudełkowy":
        method_info = get_visualization_method_description("boxplot")
        section_header(method_info["title"], "Wizualizacja rozkładu danych i wartości odstających")

        with st.expander("ℹ️ Jak interpretować wykres pudełkowy?"):
            st.markdown(method_info["description"])

        # Wybór kolumny
        column = st.selectbox(
            "Wybierz kolumnę (opcjonalne):",
            ["Wszystkie kolumny numeryczne"] + numeric_cols,
            help="Wybierz jedną kolumnę lub pokaż wszystkie kolumny numeryczne"
        )

        # Opcja grupowania według klasy
        by_class = st.checkbox(
            "Grupuj według klasy",
            value=True,
            help="Pokaż oddzielne wykresy pudełkowe dla każdej klasy wina"
        )

        # Tworzenie wykresu pudełkowego
        if column == "Wszystkie kolumny numeryczne":
            fig = create_boxplot(st.session_state.data, None, by_class)
        else:
            # Pokaż opis wybranej cechy
            show_feature_description(column)
            fig = create_boxplot(st.session_state.data, column, by_class)

        st.pyplot(fig)

        # Wskazówki interpretacji
        with st.expander("💡 Wskazówki interpretacji"):
            st.markdown("""
            **Co można odczytać z wykresu pudełkowego:**

            - **Mediana (środkowa linia)**: Wartość środkowa, dzieląca dane na dwie równe części
            - **Pudełko (IQR)**: Zawiera 50% danych, od Q1 (25%) do Q3 (75%)
            - **Wąsy**: Rozciągają się do min/max wartości w zakresie 1.5*IQR
            - **Punkty poza wąsami**: Potencjalne wartości odstające

            **Porównanie klas:**
            - Różnice w medianie między klasami wskazują na inne typowe wartości
            - Różnice w rozmiarze pudełka wskazują na różne rozproszenie danych
            - Przesunięcie pudełka w górę lub dół pokazuje różnice w rozkładzie wartości
            """)

    # Wykres rozproszenia
    elif visualization_type == "Wykres rozproszenia":
        method_info = get_visualization_method_description("scatter")
        section_header(method_info["title"], "Wizualizacja zależności między dwiema cechami")

        with st.expander("ℹ️ Jak interpretować wykres rozproszenia?"):
            st.markdown(method_info["description"])

        # Wybór kolumn
        col1, col2 = st.columns(2)

        with col1:
            x_column = st.selectbox(
                "Wybierz kolumnę dla osi X:",
                numeric_cols,
                help="Wybierz cechę do wyświetlenia na osi poziomej"
            )
            show_feature_description(x_column)

        with col2:
            # Wykluczamy kolumnę już wybraną dla X
            y_options = [col for col in numeric_cols if col != x_column]
            y_column = st.selectbox(
                "Wybierz kolumnę dla osi Y:",
                y_options,
                help="Wybierz cechę do wyświetlenia na osi pionowej"
            )
            show_feature_description(y_column)

        # Opcja kolorowania według klasy
        color_by_class = st.checkbox(
            "Koloruj według klasy",
            value=True,
            help="Użyj różnych kolorów dla punktów reprezentujących różne klasy win"
        )

        # Tworzenie wykresu rozproszenia
        if x_column and y_column:
            fig = create_scatter_plot(st.session_state.data, x_column, y_column, color_by_class)
            st.pyplot(fig)

            # Obliczanie korelacji
            corr = st.session_state.data[[x_column, y_column]].corr().iloc[0, 1]

            # Wskazówki interpretacji
            with st.expander("💡 Wskazówki interpretacji"):
                st.markdown(f"""
                **Co można odczytać z wykresu rozproszenia:**

                - **Korelacja**: Współczynnik korelacji Pearsona między {x_column} i {y_column} wynosi {corr:.2f}
                - **Wzorzec**: {'Punkty układają się wzdłuż linii, co wskazuje na silną zależność liniową' if abs(corr) > 0.7 else 'Punkty nie układają się wzdłuż linii, co sugeruje słabszą zależność liniową lub jej brak'}
                - **Skupiska**: Przyjrzyj się, czy punkty grupują się w skupiska, co może wskazywać na podgrupy w danych
                - **Wartości odstające**: Punkty znacznie oddalone od głównego skupiska mogą być wartościami odstającymi

                **Porównanie klas:**
                - Jeśli punkty różnych klas tworzą oddzielne skupiska, cecha ta może być przydatna do klasyfikacji
                - Jeśli punkty różnych klas są wymieszane, te dwie cechy mogą nie wystarczyć do rozróżnienia klas
                """)

    # Wykres rozproszenia 3D
    elif visualization_type == "Wykres rozproszenia 3D":
        method_info = get_visualization_method_description("scatter3d")
        section_header(method_info["title"], "Wizualizacja zależności między trzema cechami")

        with st.expander("ℹ️ Jak interpretować wykres rozproszenia 3D?"):
            st.markdown(method_info["description"])

        # Wybór kolumn
        col1, col2, col3 = st.columns(3)

        with col1:
            x_column = st.selectbox(
                "Wybierz kolumnę dla osi X:",
                numeric_cols,
                help="Wybierz cechę do wyświetlenia na osi X"
            )

        with col2:
            # Wykluczamy kolumnę już wybraną dla X
            y_options = [col for col in numeric_cols if col != x_column]
            y_column = st.selectbox(
                "Wybierz kolumnę dla osi Y:",
                y_options,
                help="Wybierz cechę do wyświetlenia na osi Y"
            )

        with col3:
            # Wykluczamy kolumny już wybrane dla X i Y
            z_options = [col for col in numeric_cols if col != x_column and col != y_column]
            z_column = st.selectbox(
                "Wybierz kolumnę dla osi Z:",
                z_options,
                help="Wybierz cechę do wyświetlenia na osi Z"
            )

        # Opcja kolorowania według klasy
        color_by_class = st.checkbox(
            "Koloruj według klasy",
            value=True,
            help="Użyj różnych kolorów dla punktów reprezentujących różne klasy win"
        )

        # Tworzenie wykresu rozproszenia 3D
        if x_column and y_column and z_column:
            fig = create_3d_scatter(st.session_state.data, x_column, y_column, z_column, color_by_class)
            st.pyplot(fig)

            # Wskazówki interpretacji
            with st.expander("💡 Wskazówki interpretacji"):
                st.markdown("""
                **Co można odczytać z wykresu rozproszenia 3D:**

                - **Trójwymiarowe skupiska**: Punkty mogą formować skupiska, które są trudne do zauważenia na wykresach 2D
                - **Separacja klas**: Sprawdź, czy punkty różnych klas są dobrze odseparowane w przestrzeni 3D
                - **Wzorce przestrzenne**: Szukaj nietypowych wzorców lub struktur w rozmieszczeniu punktów

                **Porady:**
                - Obróć wykres, aby zobaczyć dane z różnych perspektyw
                - Kombinacja trzech odpowiednich cech może lepiej rozdzielać klasy niż dwie cechy
                - Jeśli punkty różnych klas są dobrze odseparowane, te trzy cechy mogą być dobrymi predyktorami
                """)

    # Macierz korelacji
    elif visualization_type == "Macierz korelacji":
        method_info = get_visualization_method_description("correlation")
        section_header(method_info["title"], "Wizualizacja korelacji między wszystkimi parami cech")

        with st.expander("ℹ️ Jak interpretować macierz korelacji?"):
            st.markdown(method_info["description"])

        # Wybór kolumn
        selected_columns = st.multiselect(
            "Wybierz kolumny (opcjonalne):",
            numeric_cols,
            default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols,
            help="Wybierz kolumny do uwzględnienia w macierzy korelacji"
        )

        # Wybór metody korelacji
        method = st.selectbox(
            "Metoda korelacji:",
            ["pearson", "spearman", "kendall"],
            index=0,
            help="""
            - Pearson: mierzy liniową zależność, wymaga danych normalnych
            - Spearman: mierzy monotoniczną zależność, nie wymaga normalności
            - Kendall: podobna do Spearman, ale bardziej odporna na wartości odstające
            """
        )

        # Opcja uwzględnienia kolumny Class
        include_class = st.checkbox(
            "Uwzględnij kolumnę Class",
            value=False,
            help="Włącz, aby zobaczyć korelacje między cechami a klasą wina"
        )

        # Tworzenie macierzy korelacji
        if selected_columns:
            fig = create_correlation_heatmap(
                st.session_state.data,
                selected_columns,
                method,
                include_class
            )
            st.pyplot(fig)

            # Wskazówki interpretacji
            with st.expander("💡 Wskazówki interpretacji"):
                st.markdown("""
                **Co można odczytać z macierzy korelacji:**

                - **Silne dodatnie korelacje (bliskie 1, kolor czerwony)**: Cechy zmieniają się razem w tym samym kierunku
                - **Silne ujemne korelacje (bliskie -1, kolor niebieski)**: Cechy zmieniają się w przeciwnych kierunkach
                - **Słabe korelacje (bliskie 0, kolor biały)**: Cechy nie wykazują silnej zależności liniowej

                **Praktyczne zastosowania:**
                - Identyfikacja redundantnych cech (silnie skorelowane cechy mogą nieść podobne informacje)
                - Wybór cech do modelowania (często warto wybrać cechy słabo skorelowane ze sobą)
                - Jeśli uwzględniono klasę, sprawdź które cechy są najbardziej skorelowane z klasą
                """)

    # Wykres par cech
    elif visualization_type == "Wykres par cech":
        method_info = get_visualization_method_description("pairplot")
        section_header(method_info["title"], "Wizualizacja relacji między wieloma parami cech")

        with st.expander("ℹ️ Jak interpretować wykres par cech?"):
            st.markdown(method_info["description"])

        # Wybór maksymalnej liczby kolumn
        max_cols = st.slider(
            "Maksymalna liczba kolumn:",
            min_value=2,
            max_value=8,
            value=5,
            help="Wybierz maksymalną liczbę kolumn do uwzględnienia (więcej kolumn = dłuższy czas generowania)"
        )

        # Wybór kolumn
        available_cols = [col for col in numeric_cols if col != 'Class']
        selected_columns = st.multiselect(
            "Wybierz kolumny:",
            available_cols,
            default=available_cols[:max_cols] if len(available_cols) >= max_cols else available_cols,
            help="Wybierz kolumny do uwzględnienia w wykresie par cech"
        )

        # Wybór kolumny do kolorowania
        if 'Class' in st.session_state.data.columns:
            hue = st.selectbox(
                "Kolumna do kolorowania:",
                ["Class"] + st.session_state.data.columns.tolist(),
                index=0,
                help="Wybierz kolumnę, według której punkty będą kolorowane"
            )
        else:
            hue = st.selectbox(
                "Kolumna do kolorowania:",
                st.session_state.data.columns.tolist(),
                help="Wybierz kolumnę, według której punkty będą kolorowane"
            )

        # Informacja o czasie generowania
        if len(selected_columns) > 5:
            st.warning("Generowanie wykresu par cech dla dużej liczby kolumn może zająć więcej czasu.")

        # Tworzenie wykresu par cech
        if selected_columns:
            with st.spinner("Generowanie wykresu par cech..."):
                fig = create_pairplot(
                    st.session_state.data,
                    selected_columns,
                    hue,
                    max_cols
                )
                st.pyplot(fig)

            # Wskazówki interpretacji
            with st.expander("💡 Wskazówki interpretacji"):
                st.markdown("""
                **Co można odczytać z wykresu par cech:**

                - **Przekątna**: Histogramy pokazują rozkład każdej cechy
                - **Poza przekątną**: Wykresy rozproszenia pokazują relacje między parami cech
                - **Kolory**: Punkty są kolorowane według wybranej kolumny, co pomaga zobaczyć wzorce dla różnych grup

                **Praktyczne zastosowania:**
                - Szybkie porównanie wielu par cech jednocześnie
                - Identyfikacja cech, które dobrze rozdzielają klasy (punkty różnych kolorów są dobrze odseparowane)
                - Wykrywanie nieliniowych zależności, które mogą być mniej widoczne na pojedynczej macierzy korelacji
                """)

    # Współrzędne równoległe
    elif visualization_type == "Współrzędne równoległe":
        method_info = get_visualization_method_description("parallel")
        section_header(method_info["title"], "Wizualizacja wielu cech dla każdej obserwacji")

        with st.expander("ℹ️ Jak interpretować wykres współrzędnych równoległych?"):
            st.markdown(method_info["description"])

        # Wybór maksymalnej liczby kolumn
        max_cols = st.slider(
            "Maksymalna liczba kolumn:",
            min_value=2,
            max_value=10,
            value=6,
            help="Wybierz maksymalną liczbę kolumn do uwzględnienia"
        )

        # Wybór kolumn
        available_cols = [col for col in numeric_cols if col != 'Class']
        selected_columns = st.multiselect(
            "Wybierz kolumny:",
            available_cols,
            default=available_cols[:max_cols] if len(available_cols) >= max_cols else available_cols,
            help="Wybierz kolumny do uwzględnienia w wykresie współrzędnych równoległych"
        )

        # Wybór kolumny klasy
        class_column = st.selectbox(
            "Kolumna klasy:",
            ["Class"] if 'Class' in st.session_state.data.columns else st.session_state.data.columns.tolist(),
            help="Wybierz kolumnę określającą klasę/grupę dla każdej obserwacji"
        )

        # Tworzenie wykresu współrzędnych równoległych
        if selected_columns and class_column:
            fig = create_parallel_coordinates(
                st.session_state.data,
                selected_columns,
                class_column,
                max_cols
            )
            st.pyplot(fig)

            # Wskazówki interpretacji
            with st.expander("💡 Wskazówki interpretacji"):
                st.markdown("""
                **Co można odczytać z wykresu współrzędnych równoległych:**

                - **Linie**: Każda linia to jedna obserwacja (wino), przebiegająca przez wszystkie osie
                - **Osie pionowe**: Każda oś reprezentuje jedną cechę, ze skalą od min do max wartości
                - **Kolory**: Linie są kolorowane według wybranej kolumny klasy

                **Praktyczne zastosowania:**
                - Identyfikacja cech, które dobrze rozdzielają klasy (linie różnych kolorów przechodzą przez różne obszary osi)
                - Wykrywanie wzorców w danych wielowymiarowych
                - Identyfikacja podobnych obserwacji (linie przebiegające podobnymi ścieżkami)
                - Wykrywanie wartości odstających (linie przebiegające nietypowymi ścieżkami)
                """)

    # Rozkład klas
    elif visualization_type == "Rozkład klas":
        section_header("Rozkład klas", "Wizualizacja liczby obserwacji w każdej klasie")

        with st.expander("ℹ️ O rozkładzie klas"):
            st.markdown("""
            Wykres rozkładu klas pokazuje, ile próbek należy do każdej klasy wina.
            Zrównoważony rozkład klas jest istotny przy modelowaniu uczenia maszynowego,
            ponieważ niezrównoważone klasy mogą prowadzić do modeli stronniczych.

            W zbiorze danych Wine występują trzy klasy odpowiadające trzem różnym
            odmianom winogron z tego samego regionu Włoch.
            """)

        # Sprawdź czy kolumna 'Class' istnieje
        if 'Class' in st.session_state.data.columns:
            fig = create_class_distribution(st.session_state.data)
            st.pyplot(fig)

            # Obliczanie dokładnych wartości
            class_counts = st.session_state.data['Class'].value_counts().sort_index()
            total = len(st.session_state.data)

            # Wyświetlanie informacji o rozkładzie
            st.markdown("### Szczegóły rozkładu klas")

            for cls, count in class_counts.items():
                percentage = (count / total) * 100
                st.markdown(f"- **Klasa {cls}**: {count} próbek ({percentage:.1f}%)")

            # Ocena zrównoważenia klas
            min_count = class_counts.min()
            max_count = class_counts.max()
            ratio = min_count / max_count

            st.markdown("### Ocena zrównoważenia klas")

            if ratio > 0.8:
                st.success(f"Klasy są dobrze zrównoważone (stosunek najmniejszej do największej klasy: {ratio:.2f})")
            elif ratio > 0.5:
                st.info(f"Klasy są umiarkowanie zrównoważone (stosunek: {ratio:.2f})")
            else:
                st.warning(
                    f"Klasy są niezrównoważone (stosunek: {ratio:.2f}). Rozważ techniki jak ważenie klas lub oversampling.")
        else:
            st.error("Kolumna 'Class' nie istnieje w danych.")
