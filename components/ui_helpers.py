"""
Pomocnicze funkcje dla interfejsu użytkownika.
"""

import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt


def set_page_config():
    """Konfiguruje stronę Streamlit."""
    st.set_page_config(
        page_title="Wine Dataset Analysis",
        page_icon="🍷",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dodajemy dodatkowe CSS dla lepszego wyglądu
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #722F37;
    }
    .stButton button {
        background-color: #722F37;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #722F37;
    }
    </style>
    """, unsafe_allow_html=True)


def show_info_box(title, content):
    """
    Wyświetla box informacyjny z tytułem i treścią z użyciem czystego HTML.

    Args:
        title: Tytuł boksu informacyjnego
        content: Treść boksu informacyjnego
    """

    # Przetwarzanie treści - najpierw zastąpienie gwiazdek znacznikami HTML
    content_formatted = content

    # Zamień listę punktowaną na odpowiednie HTML
    content_formatted = content_formatted.replace("1. ", "<li><strong>")
    content_formatted = content_formatted.replace("2. ", "<li><strong>")
    content_formatted = content_formatted.replace("3. ", "<li><strong>")
    content_formatted = content_formatted.replace("- ", "<li>")

    # Dodaj zamknięcie tagów strong dla elementów listy numerowanej
    if "<li><strong>" in content_formatted:
        lines = content_formatted.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("<li><strong>"):
                # Znajdź pierwszą kropkę po numerze i zamień ją na zamknięcie tagu strong
                dot_index = line.find(":")
                if dot_index != -1:
                    lines[i] = line[:dot_index] + ":</strong>" + line[dot_index + 1:]
        content_formatted = '\n'.join(lines)

    # Zamień nagłówki i znaczniki formatujące
    content_formatted = content_formatted.replace("**Kluczowe wnioski:**", "<strong>Kluczowe wnioski:</strong>")
    content_formatted = content_formatted.replace("**Zalecenia:**", "<strong>Zalecenia:</strong>")

    # Otwórz-zamknij odpowiednio listę
    if "<li>" in content_formatted:
        content_formatted = "<ul style='margin-top: 10px; margin-bottom: 10px; padding-left: 20px;'>\n" + content_formatted + "\n</ul>"

    # Popraw formatowanie paragrafów
    content_formatted = content_formatted.replace("\n", "<br>")

    # Usuń niepotrzebny znacznik </p> jeśli się pojawia
    content_formatted = content_formatted.replace("</p>", "")

    st.markdown(f"""
    <div style="padding: 1.2rem; border-radius: 0.5rem; background-color: #1e1e1e; margin-bottom: 1rem; border: 1px solid #333333;">
        <h4 style="color: #e67e22; margin-top: 0; margin-bottom: 1rem;">{title}</h4>
        <div style="color: #f0f0f0; line-height: 1.5;">
            {content_formatted}
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_feature_description(feature_name):
    """
    Wyświetla opis danej cechy zbioru danych.

    Args:
        feature_name: Nazwa cechy z Wine Dataset
    """
    descriptions = {
        "Class": "Klasa wina (1, 2, 3) - odpowiada trzem różnym odmianom winogron/pochodzeniu",
        "Alcohol": "Zawartość alkoholu (% obj.)",
        "Malic acid": "Zawartość kwasu jabłkowego (g/l)",
        "Ash": "Zawartość popiołu (g/l) - minerałów nieorganicznych",
        "Alcalinity of ash": "Alkaliczność popiołu (pH)",
        "Magnesium": "Zawartość magnezu (mg/l)",
        "Total phenols": "Całkowita zawartość fenoli (mg/l)",
        "Flavanoids": "Zawartość flawonoidów (mg/l)",
        "Nonflavanoid phenols": "Zawartość fenoli niebędących flawonoidami (mg/l)",
        "Proanthocyanins": "Zawartość proantocyjanidyn (mg/l)",
        "Color intensity": "Intensywność koloru (absorbancja)",
        "Hue": "Odcień (wskaźnik)",
        "OD280/OD315 of diluted wines": "Stosunek absorbancji w długościach fal 280nm do 315nm (miara białek)",
        "Proline": "Zawartość proliny (aminokwasu) (mg/l)"
    }

    if feature_name in descriptions:
        with st.expander(f"ℹ️ Co oznacza {feature_name}?"):
            st.markdown(descriptions[feature_name])


def display_metric_group(metrics_dict):
    """
    Wyświetla grupę wskaźników (metryk) w układzie kolumn.

    Args:
        metrics_dict: Słownik z nazwami i wartościami metryk
    """
    cols = st.columns(len(metrics_dict))

    for i, (title, value) in enumerate(metrics_dict.items()):
        cols[i].metric(title, value)


def create_download_link(df, filename="data.csv", link_text="Pobierz dane jako CSV"):
    """
    Tworzy link do pobrania DataFrame jako plik CSV.

    Args:
        df: DataFrame do pobrania
        filename: Nazwa pliku do pobrania
        link_text: Tekst linku

    Returns:
        str: HTML z linkiem do pobrania
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def add_spaces(n=1):
    """
    Dodaje pionowe odstępy w interfejsie.

    Args:
        n: Liczba odstępów do dodania
    """
    for _ in range(n):
        st.markdown("&nbsp;", unsafe_allow_html=True)


def local_css(file_name):
    """
    Ładuje lokalny plik CSS.

    Args:
        file_name: Nazwa pliku CSS do załadowania
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def section_header(title, description=None):
    """
    Tworzy nagłówek sekcji z opisem.

    Args:
        title: Tytuł sekcji
        description: Opis sekcji
    """
    st.markdown(f"## {title}")
    if description:
        st.markdown(f"*{description}*")
    st.markdown("---")


def plot_to_html(fig):
    """
    Konwertuje obiekt Figure na kod HTML.

    Args:
        fig: Obiekt Figure z matplotlib

    Returns:
        str: HTML z osadzonym obrazem
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_str}" width="100%">'