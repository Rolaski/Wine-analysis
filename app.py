"""
Wine Dataset Analysis - Aplikacja do analizy i eksploracji zbioru danych Wine Dataset z UCI
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
from src.data_loader import load_wine_dataset
from src.utils import get_sample_wine_data

# Konfiguracja strony
set_page_config()

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
page = create_sidebar()

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