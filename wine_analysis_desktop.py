"""
Wrapper do uruchamiania aplikacji Wine Analysis jako aplikacji desktopowej.
"""

import os
import sys
import subprocess
import streamlit.web.bootstrap as bootstrap
from streamlit_desktop import desktop
import webbrowser
from pathlib import Path
import time

# Konfiguracja aplikacji
APP_NAME = "Wine Analysis"
APP_ICON = os.path.join("data", "wine_icon.ico")  # Ścieżka do ikony
APP_WIDTH = 1200  # Szerokość okna aplikacji
APP_HEIGHT = 800  # Wysokość okna aplikacji


def main():
    # Ścieżka do głównego pliku aplikacji
    app_path = os.path.abspath("app.py")

    # Sprawdź, czy plik aplikacji istnieje
    if not os.path.exists(app_path):
        print(f"Błąd: Nie znaleziono pliku {app_path}")
        input("Naciśnij Enter, aby zakończyć...")
        sys.exit(1)

    print(f"Uruchamianie aplikacji {APP_NAME}...")

    # Przygotuj argumenty dla Streamlit
    args = []

    # Uruchom aplikację w trybie desktopowym
    desktop.run(
        main_script_path=app_path,
        app_name=APP_NAME,
        window_size=(APP_WIDTH, APP_HEIGHT),
        icon_path=os.path.abspath(APP_ICON) if os.path.exists(APP_ICON) else None,
        args=args
    )


if __name__ == "__main__":
    main()