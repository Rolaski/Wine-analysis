"""
Skrypt do instalacji i konfiguracji streamlit-desktop
"""

import subprocess
import sys
import os

def install_streamlit_desktop():
    print("Instalacja streamlit-desktop...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-desktop"])
    print("Instalacja zakończona pomyślnie!")

if __name__ == "__main__":
    install_streamlit_desktop()