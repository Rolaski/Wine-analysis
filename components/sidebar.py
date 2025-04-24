"""
Komponent odpowiedzialny za pasek boczny (sidebar) w aplikacji.
"""

import streamlit as st

def create_sidebar():
    """
    Tworzy pasek boczny z nawigacją i dodatkowymi opcjami.

    Returns:
        str: Nazwa wybranej strony
    """
    # Ukryj domyślne linki w sidebarze
    hide_streamlit_links = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none}
    div[data-testid="stSidebarNav"] {display: none}
    </style>
    """
    st.markdown(hide_streamlit_links, unsafe_allow_html=True)

    # informacje o aplikacji
    st.sidebar.title("Nawigacja")

    # Menu nawigacyjne
    page = st.sidebar.radio(
        "Wybierz stronę:",
        ["Przegląd danych", "Analiza statystyczna", "Manipulacja danymi", "Wizualizacja", "Modelowanie ML"],
        help="Wybierz jedną z dostępnych stron aplikacji"
    )

    # Dodatkowe informacje w sidebarze
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Informacje o projekcie")
    st.sidebar.markdown("""
    - Dataset: [Wine Dataset (UCI)](https://archive.ics.uci.edu/dataset/109/wine)
    - Liczba próbek: 178
    - Liczba klas: 3
    - Liczba atrybutów: 13
    """)

    # Dodatkowe opcje
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Opcje")

    # Opcja zresetowania danych
    if st.sidebar.button("Zresetuj dane"):
        st.session_state.data = st.session_state.original_data.copy()
        st.sidebar.success("Dane zostały zresetowane!")

    # Pomoc
    with st.sidebar.expander("Pomoc"):
        st.markdown("""
        **Jak korzystać z aplikacji:**
        1. Wybierz interesującą Cię stronę w menu powyżej
        2. Eksploruj dane i wykonuj analizy
        3. Użyj przycisku 'Zresetuj dane', aby powrócić do oryginalnego zbioru danych
        
        Każda sekcja zawiera szczegółowe opisy dostępnych funkcji.
        """)

    return page