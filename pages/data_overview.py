"""
Moduł odpowiedzialny za stronę przeglądu danych w aplikacji Wine Dataset Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modułów własnych
from src.data_loader import get_dataset_info
from src.data_visualizer import create_class_distribution, create_correlation_heatmap
from components.descriptions import get_page_description
from components.ui_helpers import show_info_box, display_metric_group, create_download_link, section_header


def page_data_overview():
    """Wyświetla stronę przeglądu danych."""

    # Pobierz opis strony
    page_info = get_page_description("overview")

    # Nagłówek strony
    st.header(page_info["title"])
    st.markdown(page_info["description"])

    # Informacje o zbiorze danych
    section_header("Informacje o zbiorze danych",
                   "Podstawowe statystyki i parametry zbioru danych Wine Dataset")

    # Link do datasetu
    st.markdown(
        "[🔗 Link do oryginalnego zbioru danych (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/109/wine)")

    # Pobierz informacje o zbiorze danych
    info = get_dataset_info(st.session_state.data)

    # Wyświetl metryki w trzech kolumnach
    metrics = {
        "Liczba wierszy": info["liczba_wierszy"],
        "Liczba kolumn": info["liczba_kolumn"],
        "Liczba klas": info["liczba_klas"],
        "Brakujące wartości": info["brakujące_wartości"],
        "Duplikaty": info["duplikaty"]
    }
    display_metric_group(metrics)

    # Rozkład klas z opisem
    st.subheader("Rozkład klas")
    with st.expander("ℹ️ Co oznaczają klasy?", expanded=True):
        st.markdown("""
        W zbiorze danych Wine występują trzy klasy oznaczone liczbami 1, 2, 3:
        - **Klasa 1**: Wina z pierwszej odmiany winogron
        - **Klasa 2**: Wina z drugiej odmiany winogron
        - **Klasa 3**: Wina z trzeciej odmiany winogron

        Klasyfikacja oparta jest na analizie chemicznej i odpowiada różnym odmianom winogron uprawianych w tym samym 
        regionie Włoch. Celem analizy jest określenie pochodzenia wina na podstawie jego składu chemicznego.
        """)

    # Wykres rozkładu klas
    fig = create_class_distribution(st.session_state.data)
    st.pyplot(fig)

    # Opis kolumn
    st.subheader("Opis kolumn w zbiorze danych")
    with st.expander("📊 O cechach chemicznych win", expanded=True):
        st.markdown("""
        Zbiór danych zawiera 13 cech chemicznych win, które są używane do przewidywania ich pochodzenia:

        1. **Alcohol** - Zawartość alkoholu (% objętości)
        2. **Malic acid** - Zawartość kwasu jabłkowego (g/l)
        3. **Ash** - Zawartość popiołu, który reprezentuje minerały nieorganiczne (g/l)
        4. **Alcalinity of ash** - Alkaliczność popiołu, która mierzy pH minerałów
        5. **Magnesium** - Zawartość magnezu (mg/l)
        6. **Total phenols** - Całkowita zawartość fenoli, związków wpływających na smak i aromat (mg/l)
        7. **Flavanoids** - Zawartość flawonoidów, podgrupy fenoli o właściwościach antyoksydacyjnych (mg/l)
        8. **Nonflavanoid phenols** - Zawartość fenoli niebędących flawonoidami (mg/l)
        9. **Proanthocyanins** - Zawartość proantocyjanidyn, związków wpływających na barwę i cierpkość (mg/l)
        10. **Color intensity** - Intensywność koloru wina (metoda spektrofotometryczna)
        11. **Hue** - Odcień koloru wina (wskaźnik)
        12. **OD280/OD315** - Stosunek absorbancji przy długościach fal 280nm do 315nm (miara zawartości białek)
        13. **Proline** - Zawartość proliny, aminokwasu często występującego w winach (mg/l)

        Te cechy są powszechnie używane w enologii (nauce o winie) do charakteryzowania i klasyfikowania win.
        """)

    # Tabela opisu kolumn
    col_descriptions = {
        "Class": "Klasa wina (1, 2, 3) - odpowiada trzem różnym odmianom winogron/pochodzeniu",
        "Alcohol": "Zawartość alkoholu (% objętości)",
        "Malic acid": "Zawartość kwasu jabłkowego (g/l)",
        "Ash": "Zawartość popiołu (g/l) - minerały nieorganiczne",
        "Alcalinity of ash": "Alkaliczność popiołu (pH)",
        "Magnesium": "Zawartość magnezu (mg/l)",
        "Total phenols": "Całkowita zawartość fenoli (mg/l)",
        "Flavanoids": "Zawartość flawonoidów (mg/l)",
        "Nonflavanoid phenols": "Zawartość fenoli niebędących flawonoidami (mg/l)",
        "Proanthocyanins": "Zawartość proantocyjanidyn (mg/l)",
        "Color intensity": "Intensywność koloru (absorbancja)",
        "Hue": "Odcień (wskaźnik)",
        "OD280/OD315 of diluted wines": "Stosunek absorbancji 280/315nm (miara białek)",
        "Proline": "Zawartość proliny (aminokwasu) (mg/l)"
    }

    desc_df = pd.DataFrame({
        "Kolumna": col_descriptions.keys(),
        "Opis": col_descriptions.values()
    })

    st.dataframe(desc_df, use_container_width=True)

    # Próbka danych z opisem
    st.subheader("Próbka danych")
    with st.expander("ℹ️ O danych", expanded=True):
        st.markdown("""
        Poniżej przedstawiona jest próbka danych z naszego zbioru. Możesz zmienić liczbę wyświetlanych wierszy 
        za pomocą suwaka poniżej. Każdy wiersz reprezentuje pojedynczą próbkę wina, a kolumny zawierają 
        wartości poszczególnych cech chemicznych.
        """)

    # Suwak do wyboru liczby wierszy
    sample_size = st.slider("Liczba wierszy do wyświetlenia:", 5, 50, 10)

    # Wyświetlenie próbki danych z możliwością pobrania
    st.dataframe(st.session_state.data.head(sample_size), use_container_width=True)
    st.markdown(
        create_download_link(st.session_state.data, "wine_dataset.csv", "📥 Pobierz pełny zbiór danych jako CSV"),
        unsafe_allow_html=True)

    # Podstawowe statystyki
    st.subheader("Podstawowe statystyki")
    with st.expander("ℹ️ O statystykach", expanded=True):
        st.markdown("""
        Poniższa tabela zawiera podstawowe statystyki opisowe dla wszystkich cech numerycznych w zbiorze danych:
        - **count**: liczba niepustych obserwacji
        - **mean**: średnia arytmetyczna
        - **std**: odchylenie standardowe
        - **min**: wartość minimalna
        - **25%**: pierwszy kwartyl (25% obserwacji ma wartość mniejszą)
        - **50%**: mediana (wartość środkowa)
        - **75%**: trzeci kwartyl (75% obserwacji ma wartość mniejszą)
        - **max**: wartość maksymalna

        Statystyki te dają ogólny obraz rozkładu wartości w danych.
        """)

    # Wyświetlenie statystyk
    st.dataframe(st.session_state.data.describe().round(2), use_container_width=True)

    # Typy danych
    st.subheader("Typy danych")
    with st.expander("ℹ️ O typach danych", expanded=True):
        st.markdown("""
        Ta tabela pokazuje typy danych dla każdej kolumny. W analizie danych ważne jest zrozumienie,
        jakiego typu są dane w każdej kolumnie:
        - **int64**: liczby całkowite
        - **float64**: liczby zmiennoprzecinkowe (z częścią ułamkową)
        - **object**: zazwyczaj łańcuchy znaków lub dane mieszane
        - **bool**: wartości logiczne (True/False)
        - **datetime64**: daty i czasy

        W tym zbiorze danych wszystkie kolumny powinny być numeryczne (int64 lub float64).
        """)

    # Wyświetlenie typów danych
    st.dataframe(pd.DataFrame({'Typ danych': st.session_state.data.dtypes}), use_container_width=True)

    # Macierz korelacji z opisem
    st.subheader("Macierz korelacji")
    with st.expander("ℹ️ O korelacji", expanded=True):
        st.markdown("""
        Macierz korelacji pokazuje siłę zależności liniowej między parami cech. Wartości korelacji wahają się od -1 do 1:
        - **Korelacja bliskie 1**: silna dodatnia korelacja (obie cechy rosną razem)
        - **Korelacja bliskie -1**: silna ujemna korelacja (jedna cecha rośnie, gdy druga maleje)
        - **Korelacja bliska 0**: brak korelacji liniowej

        Silne korelacje mogą wskazywać na redundantne cechy lub istotne zależności w danych.
        """)

    # Wyświetlenie macierzy korelacji
    corr_fig = create_correlation_heatmap(st.session_state.data)
    st.pyplot(corr_fig)

    # Podsumowanie
    st.subheader("Podsumowanie")
    show_info_box("Wnioski z przeglądu danych", """
    - Zbiór danych Wine Dataset zawiera 13 cech chemicznych dla 178 win z trzech różnych odmian winogron.
    - Dane są kompletne (brak brakujących wartości) i nie zawierają duplikatów.
    - Widoczne są różnice w rozkładzie klas: klasa 1 (59 próbek), klasa 2 (71 próbek), klasa 3 (48 próbek).
    - Niektóre cechy wykazują silne korelacje, co może wskazywać na potencjalne redundancje.
    - Wszystkie cechy są liczbowe, co ułatwia analizę statystyczną i modelowanie.

    Te wstępne obserwacje będą pomocne przy dalszej analizie i modelowaniu danych.
    """)