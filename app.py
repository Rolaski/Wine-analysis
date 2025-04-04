"""
Główny plik aplikacji do analizy danych Wine Dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Dodanie ścieżki do modułów
sys.path.append(os.path.abspath('./src'))

# Importy własnych modułów
from src.data_loader import load_wine_dataset, get_dataset_info, get_sample_data
from src.data_manipulator import select_features, handle_missing_values
from src.statistical_analyzer import calculate_basic_stats, calculate_correlation_matrix
from src.data_visualizer import create_histogram, create_correlation_heatmap, create_class_distribution
from src.utils import get_column_types, convert_fig_to_html, get_sample_wine_data

# Konfiguracja strony
st.set_page_config(
    page_title="Wine Dataset Analysis",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Funkcja do ładowania danych
@st.cache_data
def load_data():
    """Ładuje dane wine dataset z pliku lub używa przykładowych danych."""
    try:
        df = load_wine_dataset("./data/wine.data")
        if df is None:
            st.warning("Nie udało się wczytać danych z pliku. Używam przykładowych danych.")
            df = get_sample_wine_data()
    except Exception as e:
        st.error(f"Wystąpił błąd podczas wczytywania danych: {e}")
        df = get_sample_wine_data()

    return df


# Nagłówek aplikacji
st.title("🍷 Wine Dataset Analysis")
st.markdown("""
Aplikacja do analizy i eksploracji zbioru danych Wine Dataset z UCI.
Wykorzystuje Python i biblioteki do analizy danych (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn).
""")

# Ładowanie danych
df = load_data()

# Zapisz dane w sesji
if 'data' not in st.session_state:
    st.session_state.data = df.copy()

# Sidebar
st.sidebar.title("Nawigacja")
page = st.sidebar.radio(
    "Wybierz stronę:",
    ["Przegląd danych", "Analiza statystyczna", "Manipulacja danymi", "Wizualizacja", "Modelowanie ML"]
)


# Funkcje dla poszczególnych stron
def page_data_overview():
    st.header("Przegląd danych Wine Dataset")

    # Informacje o zbiorze danych
    st.subheader("Informacje o zbiorze danych")
    info = get_dataset_info(st.session_state.data)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Liczba wierszy", info["liczba_wierszy"])
        st.metric("Liczba kolumn", info["liczba_kolumn"])
        st.metric("Liczba klas", info["liczba_klas"])

    with col2:
        st.metric("Brakujące wartości", info["brakujące_wartości"])
        st.metric("Duplikaty", info["duplikaty"])

    # Opis kolumn
    st.subheader("Opis kolumn w zbiorze danych")

    col_descriptions = {
        "Class": "Klasa wina (1, 2, 3) - odpowiada trzem różnym odmianom winogron/pochodzeniu",
        "Alcohol": "Zawartość alkoholu",
        "Malic acid": "Zawartość kwasu jabłkowego",
        "Ash": "Zawartość popiołu (minerałów nieorganicznych)",
        "Alcalinity of ash": "Alkaliczność popiołu",
        "Magnesium": "Zawartość magnezu",
        "Total phenols": "Całkowita zawartość fenoli",
        "Flavanoids": "Zawartość flawonoidów",
        "Nonflavanoid phenols": "Zawartość fenoli niebędących flawonoidami",
        "Proanthocyanins": "Zawartość proantocyjanidyn",
        "Color intensity": "Intensywność koloru",
        "Hue": "Odcień",
        "OD280/OD315 of diluted wines": "Stosunek absorbancji w długościach fal 280nm do 315nm (miara białek)",
        "Proline": "Zawartość proliny (aminokwasu)"
    }

    # Utwórz DataFrame z opisami kolumn do ładniejszego wyświetlenia
    desc_df = pd.DataFrame({
        "Kolumna": col_descriptions.keys(),
        "Opis": col_descriptions.values()
    })

    st.dataframe(desc_df, use_container_width=True)
    # Rozkład klas
    st.subheader("Rozkład klas")
    fig = create_class_distribution(st.session_state.data)
    st.pyplot(fig)

    # Próbka danych
    st.subheader("Próbka danych")
    sample_size = st.slider("Liczba wierszy do wyświetlenia", 5, 50, 10)
    st.dataframe(st.session_state.data.head(sample_size))

    # Statystyki podstawowe
    st.subheader("Statystyki podstawowe")
    st.dataframe(st.session_state.data.describe().round(2))

    # Typy danych
    st.subheader("Typy danych")
    st.dataframe(pd.DataFrame({'Typ': st.session_state.data.dtypes}))

    # Macierz korelacji
    st.subheader("Macierz korelacji")
    corr_fig = create_correlation_heatmap(st.session_state.data)
    st.pyplot(corr_fig)


def page_statistical_analysis():
    st.header("Analiza statystyczna")

    # Importuj moduły potrzebne do analizy statystycznej
    from src.statistical_analyzer import (
        calculate_basic_stats, calculate_quartiles, calculate_correlation_matrix,
        find_highly_correlated_features, calculate_class_stats, test_normality,
        calculate_outliers
    )

    # Wybierz kolumny do analizy
    st.subheader("Wybierz kolumny do analizy")
    column_types = get_column_types(st.session_state.data)

    all_columns = column_types.get('numeric', [])
    if 'Class' in all_columns:
        all_columns.remove('Class')

    selected_columns = st.multiselect(
        "Wybierz kolumny numeryczne do analizy:",
        all_columns,
        default=all_columns[:5] if len(all_columns) > 5 else all_columns
    )

    if not selected_columns:
        st.warning("Wybierz co najmniej jedną kolumnę do analizy.")
        return

    # Podstawowe statystyki
    st.subheader("Podstawowe statystyki")
    basic_stats = calculate_basic_stats(st.session_state.data, selected_columns)
    st.dataframe(basic_stats.round(3))

    # Kwartyle i percentyle
    st.subheader("Kwartyle i percentyle")
    quartiles = calculate_quartiles(st.session_state.data, selected_columns)
    st.dataframe(quartiles.round(3))

    # Test normalności
    st.subheader("Test normalności (Shapiro-Wilk)")
    normality = test_normality(st.session_state.data, selected_columns)
    st.dataframe(normality.round(4))

    # Wykresy dla wybranych kolumn
    st.subheader("Wykresy dla wybranych kolumn")

    # Wybierz kolumnę do wizualizacji
    column_to_plot = st.selectbox("Wybierz kolumnę do wizualizacji:", selected_columns)

    # Histogram
    st.subheader(f"Histogram dla {column_to_plot}")
    by_class = st.checkbox("Grupuj według klasy", value=True)
    bins = st.slider("Liczba przedziałów", 5, 50, 20)

    from src.data_visualizer import create_histogram, create_boxplot

    hist_fig = create_histogram(st.session_state.data, column_to_plot, bins, by_class)
    st.pyplot(hist_fig)

    # Wykres pudełkowy
    st.subheader(f"Wykres pudełkowy dla {column_to_plot}")
    box_fig = create_boxplot(st.session_state.data, column_to_plot, by_class=True)
    st.pyplot(box_fig)

    # Analiza korelacji
    st.subheader("Analiza korelacji")

    corr_method = st.selectbox(
        "Wybierz metodę korelacji:",
        ["pearson", "spearman", "kendall"],
        index=0
    )

    corr_threshold = st.slider(
        "Próg korelacji dla par cech:",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    # Macierz korelacji
    st.subheader(f"Macierz korelacji ({corr_method})")
    corr_matrix = calculate_correlation_matrix(
        st.session_state.data,
        method=corr_method,
        columns=selected_columns
    )
    st.dataframe(corr_matrix.round(2))

    # Pary cech o wysokiej korelacji
    st.subheader(f"Pary cech o wysokiej korelacji (>{corr_threshold})")
    high_corr = find_highly_correlated_features(
        st.session_state.data,
        threshold=corr_threshold,
        method=corr_method
    )

    if high_corr is not None and not high_corr.empty:
        st.dataframe(high_corr.round(3))
    else:
        st.info(f"Brak par cech o korelacji większej niż {corr_threshold}.")

    # Statystyki według klas
    st.subheader("Statystyki według klas")

    class_stats = calculate_class_stats(st.session_state.data, selected_columns)

    class_to_show = st.selectbox(
        "Wybierz klasę:",
        sorted(class_stats.keys())
    )

    if class_to_show in class_stats:
        st.dataframe(class_stats[class_to_show].round(3))

    # Analiza wartości odstających
    st.subheader("Analiza wartości odstających")

    outlier_method = st.selectbox(
        "Wybierz metodę wykrywania wartości odstających:",
        ["iqr", "zscore"],
        index=0
    )

    outlier_col = st.selectbox(
        "Wybierz kolumnę do analizy wartości odstających:",
        selected_columns
    )

    outliers = calculate_outliers(
        st.session_state.data,
        [outlier_col],
        method=outlier_method
    )

    if outlier_col in outliers and not outliers[outlier_col].empty:
        st.write(f"Znaleziono {len(outliers[outlier_col])} wartości odstających dla {outlier_col}:")
        st.dataframe(outliers[outlier_col])
    else:
        st.info(f"Nie znaleziono wartości odstających dla {outlier_col} metodą {outlier_method}.")


def page_data_manipulation():
    st.header("Manipulacja danymi")

    # Importuj moduły do manipulacji danymi
    from src.data_manipulator import (
        select_features, select_rows_by_class, replace_values,
        handle_missing_values, remove_duplicates, scale_data,
        encode_class, add_polynomial_features
    )

    # Podgląd aktualnych danych
    st.subheader("Aktualne dane")
    st.dataframe(st.session_state.data.head())

    # Menu operacji
    st.subheader("Wybierz operację")
    operation = st.selectbox(
        "Operacja:",
        ["Wybierz cechy", "Wybierz wiersze według klasy", "Zastąp wartości",
         "Obsłuż brakujące wartości", "Usuń duplikaty", "Skaluj dane",
         "Kodowanie binarne klasy", "Dodaj cechy wielomianowe", "Resetuj dane"]
    )

    # Wykonaj wybraną operację
    if operation == "Wybierz cechy":
        st.subheader("Wybierz cechy")

        all_columns = st.session_state.data.columns.tolist()
        selected_features = st.multiselect(
            "Wybierz kolumny do zachowania:",
            all_columns,
            default=all_columns
        )

        if st.button("Zastosuj"):
            if selected_features:
                st.session_state.data = select_features(st.session_state.data, selected_features)
                st.success(f"Wybrano {len(selected_features)} kolumn.")
            else:
                st.error("Wybierz co najmniej jedną kolumnę.")

    elif operation == "Wybierz wiersze według klasy":
        st.subheader("Wybierz wiersze według klasy")

        if 'Class' in st.session_state.data.columns:
            available_classes = sorted(st.session_state.data['Class'].unique())
            selected_classes = st.multiselect(
                "Wybierz klasy do zachowania:",
                available_classes,
                default=available_classes
            )

            if st.button("Zastosuj"):
                if selected_classes:
                    st.session_state.data = select_rows_by_class(st.session_state.data, selected_classes)
                    st.success(f"Wybrano wiersze dla klas: {selected_classes}")
                else:
                    st.error("Wybierz co najmniej jedną klasę.")
        else:
            st.error("Kolumna 'Class' nie istnieje w danych.")

    elif operation == "Zastąp wartości":
        st.subheader("Zastąp wartości")

        column = st.selectbox("Wybierz kolumnę:", st.session_state.data.columns.tolist())

        if column:
            col_type = st.session_state.data[column].dtype

            if np.issubdtype(col_type, np.number):
                old_value = st.number_input("Stara wartość:", value=0.0, step=0.1)
                new_value_type = st.selectbox(
                    "Typ nowej wartości:",
                    ["Liczba", "NaN"]
                )

                if new_value_type == "Liczba":
                    new_value = st.number_input("Nowa wartość:", value=0.0, step=0.1)
                else:
                    new_value = None

                if st.button("Zastosuj"):
                    st.session_state.data = replace_values(
                        st.session_state.data, column, old_value, new_value
                    )
                    st.success(f"Zastąpiono wartości w kolumnie {column}.")
            else:
                old_value = st.text_input("Stara wartość:")
                new_value = st.text_input("Nowa wartość:")

                if st.button("Zastosuj"):
                    st.session_state.data = replace_values(
                        st.session_state.data, column, old_value, new_value
                    )
                    st.success(f"Zastąpiono wartości w kolumnie {column}.")

    elif operation == "Obsłuż brakujące wartości":
        st.subheader("Obsłuż brakujące wartości")

        strategy = st.selectbox(
            "Wybierz strategię:",
            ["mean", "median", "most_frequent", "constant"],
            index=0
        )

        if st.button("Zastosuj"):
            st.session_state.data = handle_missing_values(st.session_state.data, strategy)
            st.success(f"Uzupełniono brakujące wartości strategią: {strategy}.")

    elif operation == "Usuń duplikaty":
        st.subheader("Usuń duplikaty")

        if st.button("Zastosuj"):
            original_len = len(st.session_state.data)
            st.session_state.data = remove_duplicates(st.session_state.data)
            new_len = len(st.session_state.data)

            if original_len == new_len:
                st.info("Nie znaleziono duplikatów.")
            else:
                st.success(f"Usunięto {original_len - new_len} duplikatów.")

    elif operation == "Skaluj dane":
        st.subheader("Skaluj dane")

        # Wybierz metodę skalowania
        method = st.selectbox(
            "Wybierz metodę skalowania:",
            ["standard", "minmax"],
            index=0
        )

        # Wybierz kolumny do skalowania
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')

        columns_to_scale = st.multiselect(
            "Wybierz kolumny do skalowania:",
            numeric_cols,
            default=numeric_cols
        )

        if st.button("Zastosuj"):
            if columns_to_scale:
                st.session_state.data = scale_data(
                    st.session_state.data, method, columns_to_scale
                )
                st.success(f"Przeskalowano dane metodą: {method}")
            else:
                st.error("Wybierz co najmniej jedną kolumnę do skalowania.")

    elif operation == "Kodowanie binarne klasy":
        st.subheader("Kodowanie binarne klasy")

        if 'Class' in st.session_state.data.columns:
            if st.button("Zastosuj"):
                st.session_state.data = encode_class(st.session_state.data)
                st.success("Zastosowano kodowanie one-hot dla kolumny 'Class'.")
        else:
            st.error("Kolumna 'Class' nie istnieje w danych.")

    elif operation == "Dodaj cechy wielomianowe":
        st.subheader("Dodaj cechy wielomianowe")

        # Wybierz kolumny
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')

        columns_to_transform = st.multiselect(
            "Wybierz kolumny do transformacji:",
            numeric_cols
        )

        # Wybierz stopień wielomianu
        degree = st.slider("Stopień wielomianu:", 2, 5, 2)

        if st.button("Zastosuj"):
            if columns_to_transform:
                st.session_state.data = add_polynomial_features(
                    st.session_state.data, columns_to_transform, degree
                )
                st.success(f"Dodano cechy wielomianowe stopnia {degree}.")
            else:
                st.error("Wybierz co najmniej jedną kolumnę.")

    elif operation == "Resetuj dane":
        st.subheader("Resetuj dane")

        if st.button("Resetuj do oryginalnych danych"):
            st.session_state.data = load_data()
            st.success("Dane zostały zresetowane do stanu początkowego.")

    # Podgląd zmodyfikowanych danych
    st.subheader("Zmodyfikowane dane")
    st.dataframe(st.session_state.data.head())

    # Informacje o aktualnych danych
    st.subheader("Informacje o aktualnych danych")
    info = get_dataset_info(st.session_state.data)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Liczba wierszy", info["liczba_wierszy"])
        st.metric("Liczba kolumn", info["liczba_kolumn"])

    with col2:
        st.metric("Brakujące wartości", info["brakujące_wartości"])
        st.metric("Duplikaty", info["duplikaty"])


def page_visualization():
    st.header("Wizualizacja danych")

    # Importy potrzebne dla wizualizacji
    from src.data_visualizer import (
        create_histogram, create_boxplot, create_scatter_plot,
        create_correlation_heatmap, create_pairplot, create_3d_scatter,
        create_parallel_coordinates, create_class_distribution,
        create_feature_importance
    )

    # Typy wykresów
    st.subheader("Wybierz typ wykresu")

    visualization_type = st.selectbox(
        "Typ wizualizacji:",
        ["Histogram", "Wykres pudełkowy", "Wykres rozproszenia", "Wykres rozproszenia 3D",
         "Macierz korelacji", "Wykres par cech", "Współrzędne równoległe",
         "Rozkład klas"]
    )

    # Pobierz typy kolumn
    column_types = get_column_types(st.session_state.data)
    numeric_cols = column_types.get('numeric', [])

    # Histogram
    if visualization_type == "Histogram":
        st.subheader("Histogram")

        column = st.selectbox("Wybierz kolumnę:", numeric_cols)
        bins = st.slider("Liczba przedziałów:", 5, 50, 20)
        by_class = st.checkbox("Grupuj według klasy", value=True)

        if column:
            fig = create_histogram(st.session_state.data, column, bins, by_class)
            st.pyplot(fig)

    # Wykres pudełkowy
    elif visualization_type == "Wykres pudełkowy":
        st.subheader("Wykres pudełkowy")

        column = st.selectbox("Wybierz kolumnę (opcjonalne):", ["Wszystkie kolumny numeryczne"] + numeric_cols)
        by_class = st.checkbox("Grupuj według klasy", value=True)

        if column == "Wszystkie kolumny numeryczne":
            fig = create_boxplot(st.session_state.data, None, by_class)
        else:
            fig = create_boxplot(st.session_state.data, column, by_class)

        st.pyplot(fig)

    # Wykres rozproszenia
    elif visualization_type == "Wykres rozproszenia":
        st.subheader("Wykres rozproszenia")

        col1, col2 = st.columns(2)

        with col1:
            x_column = st.selectbox("Wybierz kolumnę dla osi X:", numeric_cols)

        with col2:
            # Wykluczamy kolumnę już wybraną dla X
            y_options = [col for col in numeric_cols if col != x_column]
            y_column = st.selectbox("Wybierz kolumnę dla osi Y:", y_options)

        color_by_class = st.checkbox("Koloruj według klasy", value=True)

        if x_column and y_column:
            fig = create_scatter_plot(st.session_state.data, x_column, y_column, color_by_class)
            st.pyplot(fig)

    # Wykres rozproszenia 3D
    elif visualization_type == "Wykres rozproszenia 3D":
        st.subheader("Wykres rozproszenia 3D")

        col1, col2, col3 = st.columns(3)

        with col1:
            x_column = st.selectbox("Wybierz kolumnę dla osi X:", numeric_cols)

        with col2:
            # Wykluczamy kolumnę już wybraną dla X
            y_options = [col for col in numeric_cols if col != x_column]
            y_column = st.selectbox("Wybierz kolumnę dla osi Y:", y_options)

        with col3:
            # Wykluczamy kolumny już wybrane dla X i Y
            z_options = [col for col in numeric_cols if col != x_column and col != y_column]
            z_column = st.selectbox("Wybierz kolumnę dla osi Z:", z_options)

        color_by_class = st.checkbox("Koloruj według klasy", value=True)

        if x_column and y_column and z_column:
            fig = create_3d_scatter(st.session_state.data, x_column, y_column, z_column, color_by_class)
            st.pyplot(fig)

    # Macierz korelacji
    elif visualization_type == "Macierz korelacji":
        st.subheader("Macierz korelacji")

        # Wybierz kolumny
        selected_columns = st.multiselect(
            "Wybierz kolumny (opcjonalne):",
            numeric_cols,
            default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
        )

        # Wybierz metodę korelacji
        method = st.selectbox(
            "Metoda korelacji:",
            ["pearson", "spearman", "kendall"],
            index=0
        )

        # Czy uwzględnić klasę
        include_class = st.checkbox("Uwzględnij kolumnę Class", value=False)

        if selected_columns:
            fig = create_correlation_heatmap(
                st.session_state.data,
                selected_columns,
                method,
                include_class
            )
            st.pyplot(fig)

    # Wykres par cech
    elif visualization_type == "Wykres par cech":
        st.subheader("Wykres par cech")

        # Wybierz kolumny
        max_cols = st.slider("Maksymalna liczba kolumn:", 2, 8, 5)

        available_cols = [col for col in numeric_cols if col != 'Class']
        selected_columns = st.multiselect(
            "Wybierz kolumny:",
            available_cols,
            default=available_cols[:max_cols] if len(available_cols) >= max_cols else available_cols
        )

        # Wybierz kolumnę do kolorowania
        if 'Class' in st.session_state.data.columns:
            hue = st.selectbox(
                "Kolumna do kolorowania:",
                ["Class"] + st.session_state.data.columns.tolist(),
                index=0
            )
        else:
            hue = st.selectbox(
                "Kolumna do kolorowania:",
                st.session_state.data.columns.tolist()
            )

        if selected_columns:
            fig = create_pairplot(
                st.session_state.data,
                selected_columns,
                hue,
                max_cols
            )
            st.pyplot(fig)

    # Współrzędne równoległe
    elif visualization_type == "Współrzędne równoległe":
        st.subheader("Wykres współrzędnych równoległych")

        # Wybierz kolumny
        max_cols = st.slider("Maksymalna liczba kolumn:", 2, 10, 6)

        available_cols = [col for col in numeric_cols if col != 'Class']
        selected_columns = st.multiselect(
            "Wybierz kolumny:",
            available_cols,
            default=available_cols[:max_cols] if len(available_cols) >= max_cols else available_cols
        )
        # Wybierz kolumnę klasy
        class_column = st.selectbox(
            "Kolumna klasy:",
            ["Class"] if 'Class' in st.session_state.data.columns else st.session_state.data.columns.tolist()
        )

        if selected_columns and class_column:
            fig = create_parallel_coordinates(
                st.session_state.data,
                selected_columns,
                class_column,
                max_cols
            )
            st.pyplot(fig)

    # Rozkład klas
    elif visualization_type == "Rozkład klas":
        st.subheader("Rozkład klas")

        if 'Class' in st.session_state.data.columns:
            fig = create_class_distribution(st.session_state.data)
            st.pyplot(fig)
        else:
            st.error("Kolumna 'Class' nie istnieje w danych.")


def page_ml_modeling():
    st.header("Modelowanie uczenia maszynowego")

    # Importy potrzebne dla modelowania ML
    from src.ml_modeler import ClassificationModel, ClusteringModel, AssociationRulesMiner
    from src.utils import format_classification_report, format_confusion_matrix, generate_model_parameter_description

    # Wybierz typ modelowania
    st.subheader("Wybierz typ modelowania")


    model_category = st.selectbox(
        "Kategoria modelu:",
        ["Klasyfikacja", "Klastrowanie", "Reguły asocjacyjne"]
    )

    # Klasyfikacja
    if model_category == "Klasyfikacja":
        st.subheader("Klasyfikacja")

        # Sprawdź, czy kolumna Class istnieje
        if 'Class' not in st.session_state.data.columns:
            st.error("Kolumna 'Class' nie istnieje w danych. Nie można wykonać klasyfikacji.")
            return

        # Wybierz model klasyfikacji
        model_type = st.selectbox(
            "Wybierz model klasyfikacji:",
            ["Random Forest (rf)", "K-Nearest Neighbors (knn)", "Support Vector Machine (svm)"],
            index=0
        )

        # Mapowanie wyświetlanej nazwy na kod
        model_code = model_type.lower().replace('-', '')
        # Inicjalizacja modelu - zapewniamy konkretną wartość 'kmeans' dla K-Means
        if 'kmeans' in model_code.lower().replace('-', ''):
            model = ClusteringModel('kmeans')
        else:
            model = ClusteringModel(model_code)

        # Parametry modelu
        st.subheader("Parametry modelu")

        param_desc = generate_model_parameter_description(model_code)
        params = {}

        for param_name, param_info in param_desc.items():
            if 'options' in param_info:
                params[param_name] = st.selectbox(
                    param_info['name'],
                    param_info['options'],
                    index=param_info['options'].index(param_info['default']) if param_info['default'] in param_info[
                        'options'] else 0
                )
            elif 'min' in param_info and 'max' in param_info:
                step = param_info.get('step', 1)
                params[param_name] = st.slider(
                    param_info['name'],
                    param_info['min'],
                    param_info['max'],
                    param_info['default'],
                    step
                )

        # Przygotowanie danych
        from src.data_loader import split_features_target

        X, y = split_features_target(st.session_state.data)

        # Tworzenie i trening modelu
        if st.button("Trenuj model"):
            with st.spinner("Trening modelu..."):
                # Inicjalizacja modelu
                model = ClassificationModel(model_code)

                # Przygotowanie danych
                test_size = st.slider("Proporcja zbioru testowego:", 0.1, 0.5, 0.2, 0.05)
                model.prepare_data(X, y, test_size=test_size)

                # Trening modelu
                results = model.train(params)

                # Wyświetlenie wyników
                st.subheader("Wyniki klasyfikacji")

                # Metryki
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Dokładność (zbiór treningowy)", f"{results['train_accuracy']:.3f}")

                with col2:
                    st.metric("Dokładność (zbiór testowy)", f"{results['test_accuracy']:.3f}")

                with col3:
                    st.metric("Dokładność (walidacja krzyżowa)",
                              f"{results['cross_val_mean']:.3f} ± {results['cross_val_std']:.3f}")

                # Raport klasyfikacji
                st.subheader("Raport klasyfikacji")
                report_df = format_classification_report(results['classification_report'])
                st.dataframe(report_df)

                # Macierz pomyłek
                st.subheader("Macierz pomyłek")
                classes = sorted(y.unique())
                conf_matrix_df = format_confusion_matrix(results['confusion_matrix'], [str(c) for c in classes])
                st.dataframe(conf_matrix_df)

                # Ważność cech
                if 'feature_importance' in results:
                    st.subheader("Ważność cech")

                    # Sortuj cechy według ważności
                    feature_imp = results['feature_importance']
                    sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)

                    features = [x[0] for x in sorted_features]
                    importance = [x[1] for x in sorted_features]

                    from src.data_visualizer import create_feature_importance
                    fig = create_feature_importance(features, importance)
                    st.pyplot(fig)

    # Klastrowanie
    elif model_category == "Klastrowanie":
        st.subheader("Klastrowanie")

        # Wybierz model klastrowania
        model_type = st.selectbox(
            "Wybierz model klastrowania:",
            ["K-Means", "DBSCAN"],
            index=0
        )

        # Mapowanie wyświetlanej nazwy na kod
        model_code = model_type.lower()

        # Parametry modelu
        st.subheader("Parametry modelu")

        param_desc = generate_model_parameter_description(model_code)
        params = {}

        for param_name, param_info in param_desc.items():
            if 'options' in param_info:
                params[param_name] = st.selectbox(
                    param_info['name'],
                    param_info['options'],
                    index=param_info['options'].index(param_info['default']) if param_info['default'] in param_info[
                        'options'] else 0
                )
            elif 'min' in param_info and 'max' in param_info:
                step = param_info.get('step', 1)
                params[param_name] = st.slider(
                    param_info['name'],
                    param_info['min'],
                    param_info['max'],
                    param_info['default'],
                    step
                )

        # Wybierz cechy
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')

        selected_features = st.multiselect(
            "Wybierz cechy do klastrowania:",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )

        # Tworzenie i trening modelu
        if st.button("Wykonaj klastrowanie") and selected_features:
            with st.spinner("Klastrowanie danych..."):
                # Przygotowanie danych
                X = st.session_state.data[selected_features]

                # Inicjalizacja modelu
                model = ClusteringModel(model_code)

                # Przygotowanie danych
                model.prepare_data(X)

                # Znajdowanie optymalnej liczby klastrów dla K-Means
                if model_code == 'kmeans':
                    st.subheader("Znajdowanie optymalnej liczby klastrów")
                    max_clusters = st.slider("Maksymalna liczba klastrów do sprawdzenia:", 2, 15, 10)

                    with st.spinner("Szukanie optymalnej liczby klastrów..."):
                        optimal_results = model.find_optimal_clusters(max_clusters)

                        # Wykres metody łokcia (inertia)
                        st.subheader("Metoda łokcia (inertia)")


                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(optimal_results["k_values"], optimal_results["inertias"], 'o-')
                        ax.set_xlabel('Liczba klastrów (k)')
                        ax.set_ylabel('Inertia')
                        ax.set_title('Metoda łokcia dla określenia optymalnej liczby klastrów')
                        st.pyplot(fig)

                        # Wykres współczynnika silhouette
                        st.subheader("Współczynnik silhouette")

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(optimal_results["k_values"], optimal_results["silhouettes"], 'o-')
                        ax.set_xlabel('Liczba klastrów (k)')
                        ax.set_ylabel('Współczynnik silhouette')
                        ax.set_title('Współczynnik silhouette dla określenia optymalnej liczby klastrów')
                        st.pyplot(fig)

                        # Informacja o optymalnej liczbie klastrów
                        st.info(
                            f"Optymalna liczba klastrów na podstawie współczynnika silhouette: {optimal_results['optimal_k']}")

                        # Aktualizacja liczby klastrów
                        params['n_clusters'] = optimal_results['optimal_k']

                # Trening modelu
                results = model.train(params)

                # Wyświetlenie wyników
                st.subheader("Wyniki klastrowania")

                # Podstawowe informacje
                st.write(f"Liczba klastrów: {results['n_clusters']}")

                # Liczba próbek w każdym klastrze
                st.subheader("Rozkład klastrów")
                cluster_sizes = pd.DataFrame.from_dict(results['cluster_sizes'], orient='index',
                                                       columns=['Liczba próbek'])
                st.dataframe(cluster_sizes)

                # Wykres rozkładu klastrów
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(cluster_sizes.index.astype(str), cluster_sizes['Liczba próbek'])
                ax.set_xlabel('Klaster')
                ax.set_ylabel('Liczba próbek')
                ax.set_title('Rozkład próbek w klastrach')
                st.pyplot(fig)

                # Dodatkowe informacje dla K-Means
                if model_code == 'kmeans':
                    st.subheader("Inertia")
                    st.write(f"Inertia: {results['inertia']:.3f}")

                # Współczynnik silhouette (jeśli dostępny)
                if 'silhouette_score' in results:
                    st.subheader("Współczynnik silhouette")
                    st.write(f"Współczynnik silhouette: {results['silhouette_score']:.3f}")

                # Wizualizacja klastrów na wykresie 2D
                if len(selected_features) >= 2:
                    st.subheader("Wizualizacja klastrów (2D)")

                    # Pobierz etykiety klastrów
                    cluster_labels = model.get_clusters()

                    # Wybierz dwie cechy do wizualizacji
                    feat1, feat2 = selected_features[:2]

                    # Stwórz wykres
                    fig, ax = plt.subplots(figsize=(10, 8))

                    scatter = ax.scatter(
                        X[feat1],
                        X[feat2],
                        c=cluster_labels,
                        cmap='viridis',
                        alpha=0.8,
                        s=50
                    )

                    # Dodaj centra klastrów dla K-Means
                    if model_code == 'kmeans' and 'cluster_centers' in results:
                        centers = np.array(results['cluster_centers'])
                        ax.scatter(
                            centers[:, X.columns.get_loc(feat1)],
                            centers[:, X.columns.get_loc(feat2)],
                            c='red',
                            marker='X',
                            s=200,
                            alpha=1,
                            label='Centra klastrów'
                        )
                        ax.legend()

                    ax.set_xlabel(feat1)
                    ax.set_ylabel(feat2)
                    ax.set_title(f'Wizualizacja klastrów: {feat1} vs {feat2}')

                    # Dodaj legendę z etykietami klastrów
                    legend = ax.legend(*scatter.legend_elements(), title="Klastry")
                    ax.add_artist(legend)

                    st.pyplot(fig)

                # Wizualizacja klastrów na wykresie 3D (jeśli dostępne są co najmniej 3 cechy)
                if len(selected_features) >= 3:
                    st.subheader("Wizualizacja klastrów (3D)")

                    # Wybierz trzy cechy do wizualizacji
                    feat1, feat2, feat3 = selected_features[:3]

                    # Stwórz wykres 3D
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    scatter = ax.scatter(
                        X[feat1],
                        X[feat2],
                        X[feat3],
                        c=cluster_labels,
                        cmap='viridis',
                        alpha=0.8,
                        s=50
                    )

                    # Dodaj centra klastrów dla K-Means w 3D
                    if model_code == 'kmeans' and 'cluster_centers' in results:
                        centers = np.array(results['cluster_centers'])
                        ax.scatter(
                            centers[:, X.columns.get_loc(feat1)],
                            centers[:, X.columns.get_loc(feat2)],
                            centers[:, X.columns.get_loc(feat3)],
                            c='red',
                            marker='X',
                            s=200,
                            alpha=1,
                            label='Centra klastrów'
                        )
                        ax.legend()

                    ax.set_xlabel(feat1)
                    ax.set_ylabel(feat2)
                    ax.set_zlabel(feat3)
                    ax.set_title(f'Wizualizacja klastrów 3D: {feat1} vs {feat2} vs {feat3}')

                    # Dodaj legendę z etykietami klastrów
                    legend = ax.legend(*scatter.legend_elements(), title="Klastry")
                    ax.add_artist(legend)

                    st.pyplot(fig)

        elif not selected_features and st.button("Wykonaj klastrowanie"):
            st.error("Wybierz co najmniej jedną cechę do klastrowania.")

    # Reguły asocjacyjne
    elif model_category == "Reguły asocjacyjne":
        st.subheader("Wydobywanie reguł asocjacyjnych")

        # Wybierz cechy
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        selected_features = st.multiselect(
            "Wybierz cechy do analizy:",
            numeric_cols,
            default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
        )

        # Parametry
        st.subheader("Parametry")

        threshold = st.slider(
            "Próg binaryzacji (percentyl):",
            0.0, 1.0, 0.5, 0.05,
            help="Wartości powyżej tego percentyla będą traktowane jako 1, poniżej jako 0."
        )

        min_support = st.slider(
            "Minimalne wsparcie:",
            0.01, 0.5, 0.1, 0.01,
            help="Minimalna częstość występowania zbioru elementów w danych."
        )

        min_confidence = st.slider(
            "Minimalna pewność:",
            0.5, 1.0, 0.7, 0.05,
            help="Minimalna pewność reguły (wsparcie(X,Y) / wsparcie(X))."
        )

        min_lift = st.slider(
            "Minimalny lift:",
            1.0, 5.0, 1.2, 0.1,
            help="Minimalna wartość liftu (wsparcie(X,Y) / (wsparcie(X) * wsparcie(Y)))."
        )

        metric = st.selectbox(
            "Metryka do sortowania reguł:",
            ["confidence", "lift", "leverage", "conviction"],
            index=1
        )

        # Tworzenie i trening modelu
        if st.button("Znajdź reguły asocjacyjne") and selected_features:
            with st.spinner("Wydobywanie reguł..."):
                # Przygotowanie danych
                X = st.session_state.data[selected_features]

                # Inicjalizacja modelu
                miner = AssociationRulesMiner()

                # Przygotowanie danych
                miner.prepare_data(X, threshold)

                # Znajdź częste zbiory elementów
                frequent_itemsets = miner.find_frequent_itemsets(min_support)

                # Generuj reguły
                min_threshold = min_confidence if metric == 'confidence' else min_lift
                rules = miner.generate_rules(min_threshold, metric)

                # Wyświetl wyniki
                st.subheader("Częste zbiory elementów")

                if frequent_itemsets is not None and not frequent_itemsets.empty:
                    st.write(f"Znaleziono {len(frequent_itemsets)} częstych zbiorów elementów.")
                    st.dataframe(frequent_itemsets.sort_values('support', ascending=False).head(20))
                else:
                    st.info("Nie znaleziono częstych zbiorów elementów z podanymi parametrami.")

                st.subheader("Reguły asocjacyjne")

                if rules is not None and not rules.empty:
                    st.write(f"Znaleziono {len(rules)} reguł asocjacyjnych.")

                    # Pokaż najlepsze reguły
                    top_rules = miner.get_top_rules(10, metric)
                    st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

                    # Sformatowane reguły
                    st.subheader("Najlepsze reguły (sformatowane)")
                    formatted_rules = miner.format_rules(top_rules)

                    for i, rule in enumerate(formatted_rules, 1):
                        st.write(f"{i}. {rule}")

                    # Wizualizacja reguł
                    st.subheader("Wizualizacja reguł asocjacyjnych")

                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Wybierz reguły do wizualizacji
                    viz_rules = top_rules.head(15)

                    # Stwórz etykiety dla reguł
                    rule_labels = [
                        f"{', '.join(list(r['antecedents']))} => {', '.join(list(r['consequents']))}"
                        for _, r in viz_rules.iterrows()
                    ]

                    # Skróć długie etykiety
                    rule_labels = [label[:50] + '...' if len(label) > 50 else label for label in rule_labels]

                    # Wykres wsparcia i pewności
                    ax.scatter(viz_rules['support'], viz_rules['confidence'], s=viz_rules['lift'] * 100, alpha=0.6)

                    # Dodaj etykiety dla punktów
                    for i, label in enumerate(rule_labels):
                        ax.annotate(
                            label,
                            (viz_rules['support'].iloc[i], viz_rules['confidence'].iloc[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center'
                        )

                    ax.set_xlabel('Wsparcie')
                    ax.set_ylabel('Pewność')
                    ax.set_title('Reguły asocjacyjne: wsparcie vs pewność (rozmiar = lift)')

                    st.pyplot(fig)
                else:
                    st.info("Nie znaleziono reguł asocjacyjnych z podanymi parametrami.")

        elif not selected_features and st.button("Znajdź reguły asocjacyjne"):
            st.error("Wybierz co najmniej jedną cechę do analizy.")


# Wywołaj odpowiednią funkcję strony
if page == "Przegląd danych":
    page_data_overview()
elif page == "Analiza statystyczna":
    page_statistical_analysis()
elif page == "Manipulacja danymi":
    page_data_manipulation()
elif page == "Wizualizacja":
    page_visualization()
elif page == "Modelowanie ML":
    page_ml_modeling()