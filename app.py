"""
Wine Dataset Analysis - Aplikacja do analizy i eksploracji zbioru danych Wine Dataset z UCI
Rozszerzona o możliwość wczytywania własnych danych CSV.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Dodanie ścieżek do modułów
sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('./pages'))
sys.path.append(os.path.abspath('./components'))

# Import komponentów
from components.sidebar import create_sidebar
from components.ui_helpers import set_page_config

# Import stron
from pages.data_overview import page_data_overview
from pages.statistical_analysis import page_statistical_analysis
from pages.data_manipulation import page_data_manipulation
from pages.visualization import page_visualization
from pages.ml_modeling import page_ml_modeling

# Import funkcji z src
from src.data_loader import load_wine_dataset, create_data_upload_interface, get_dataset_info
from src.utils import get_sample_wine_data

# Konfiguracja strony
set_page_config()

# Funkcja do ładowania danych
@st.cache_data
def load_default_data():
    """Ładuje domyślne dane wine dataset z pliku lub używa przykładowych danych."""
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
Aplikacja do analizy i eksploracji zbiorów danych z możliwością wczytywania własnych plików CSV.
Wykorzystuje Python i biblioteki do analizy danych (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn).
""")

# Wybór źródła danych
st.markdown("---")
st.subheader("🗂️ Źródło danych")

data_source = st.radio(
    "Wybierz źródło danych:",
    ["Wine Dataset (domyślny)", "Własny plik CSV"],
    index=0,
    help="Wybierz czy chcesz użyć domyślnego zbioru Wine Dataset czy wczytać własny plik CSV"
)

# Inicjalizacja danych w session_state
if 'data_source' not in st.session_state:
    st.session_state.data_source = "Wine Dataset (domyślny)"

# Sprawdź czy zmieniono źródło danych
if st.session_state.data_source != data_source:
    st.session_state.data_source = data_source
    # Wyczyść cache danych przy zmianie źródła
    if 'data' in st.session_state:
        del st.session_state.data
    if 'original_data' in st.session_state:
        del st.session_state.original_data

# Wczytywanie danych w zależności od wyboru
if data_source == "Wine Dataset (domyślny)":
    # Domyślny Wine Dataset
    if 'data' not in st.session_state or st.session_state.get('data_source_loaded') != 'wine':
        with st.spinner("Wczytywanie domyślnego zbioru danych Wine Dataset..."):
            df = load_default_data()
            st.session_state.data = df.copy()
            st.session_state.original_data = df.copy()
            st.session_state.data_source_loaded = 'wine'

        st.success("Wczytano domyślny zbiór danych Wine Dataset!")

        # Pokaż podstawowe informacje
        info = get_dataset_info(st.session_state.data)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wiersze", info["liczba_wierszy"])
        with col2:
            st.metric("Kolumny", info["liczba_kolumn"])
        with col3:
            st.metric("Klasy", info["liczba_klas"])
        with col4:
            st.metric("Brakujące wartości", info["brakujące_wartości"])

else:
    # Własny plik CSV
    st.markdown("### Wczytaj własne dane")

    # Interface do wczytywania CSV
    uploaded_df = create_data_upload_interface()

    if uploaded_df is not None:
        # Zastąp dane w session_state
        st.session_state.data = uploaded_df.copy()
        st.session_state.original_data = uploaded_df.copy()
        st.session_state.data_source_loaded = 'csv'

        st.balloons()  # Animacja sukcesu
        st.success("Dane zostały pomyślnie wczytane i są gotowe do analizy!")
        st.info("Możesz teraz przejść do innych sekcji aplikacji używając menu po lewej stronie.")

    elif 'data' not in st.session_state or st.session_state.get('data_source_loaded') != 'csv':
        # Jeśli nie wczytano jeszcze pliku CSV, pokaż instrukcje
        st.info("👆 Wybierz plik CSV powyżej, aby rozpocząć analizę własnych danych.")

        # Pokaż przykład oczekiwanego formatu
        with st.expander("📋 Przykład oczekiwanego formatu CSV"):
            example_data = {
                'Feature1': [1.2, 2.1, 3.0, 1.8, 2.5],
                'Feature2': [4.5, 5.2, 4.8, 5.1, 4.9],
                'Feature3': [0.8, 1.1, 0.9, 1.0, 0.7],
                'Class': ['A', 'B', 'A', 'C', 'B']
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df)
            st.caption("Przykład: dane z trzema cechami numerycznymi i kolumną klas")

        # Zatrzymaj wykonywanie reszty aplikacji jeśli nie ma danych
        st.stop()

# Sprawdź czy dane są wczytane
if 'data' not in st.session_state:
    st.error("Błąd: Nie wczytano żadnych danych. Spróbuj ponownie.")
    st.stop()

# Informacje o aktualnie wczytanych danych
with st.expander("ℹ️ Informacje o aktualnie wczytanych danych", expanded=False):
    info = get_dataset_info(st.session_state.data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Podstawowe informacje:**")
        st.write(f"- Źródło: {data_source}")
        st.write(f"- Liczba wierszy: {info['liczba_wierszy']}")
        st.write(f"- Liczba kolumn: {info['liczba_kolumn']}")
        st.write(f"- Brakujące wartości: {info['brakujące_wartości']}")
        st.write(f"- Duplikaty: {info['duplikaty']}")

    with col2:
        st.markdown("**Informacje o klasach:**")
        if info['liczba_klas'] > 0:
            st.write(f"- Liczba klas: {info['liczba_klas']}")
            st.write("- Rozkład klas:")
            for klasa, liczba in info['rozkład_klas'].items():
                st.write(f"  - Klasa {klasa}: {liczba} próbek")
        else:
            st.write("- Dane bez etykiet klas")

    # Podgląd typów danych
    st.markdown("**Typy kolumn:**")
    types_df = pd.DataFrame([
        {"Kolumna": col, "Typ": str(dtype)}
        for col, dtype in info['typy_danych'].items()
    ])
    st.dataframe(types_df, use_container_width=True)

# Sidebar z nawigacją
page = create_sidebar()

# Upewnij się, że oryginalne dane są zachowane
if 'original_data' not in st.session_state:
    st.session_state.original_data = st.session_state.data.copy()

# Wyświetl odpowiednią stronę
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

# Footer z informacjami
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2em;'>
    <p>🍷 Wine Dataset Analysis | Rozszerzona wersja z obsługą własnych danych CSV</p>
    <p>Wykorzystuje: Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn</p>
</div>
""", unsafe_allow_html=True)