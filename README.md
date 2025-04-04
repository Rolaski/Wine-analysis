# Wine Dataset Analysis

Interaktywna aplikacja do analizy i eksploracji zbioru danych Wine Dataset z UCI z wykorzystaniem Pythona i bibliotek do analizy danych (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit).

## Opis projektu

Aplikacja umożliwia:
- Odczyt i analizę danych z pliku CSV
- Wykonanie analizy statystycznej (min, max, mediana, moda, odchylenie standardowe itp.)
- Wyznaczanie korelacji między cechami
- Ekstrakcję podtablic i manipulację danymi
- Wizualizację danych (wykresy zależności, histogramy, wykresy korelacji)
- Przeprowadzenie analizy z wykorzystaniem modeli klasyfikacji, grupowania i reguł asocjacyjnych
- Interaktywną analizę danych poprzez GUI

## Struktura projektu

```
wine_analysis_app/
│
├── data/
│   ├── wine.data
│   ├── wine.names
│   └── index
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Moduł do wczytywania danych
│   ├── data_manipulator.py   # Moduł do manipulacji danymi
│   ├── statistical_analyzer.py # Moduł do analizy statystycznej
│   ├── data_visualizer.py    # Moduł do wizualizacji
│   ├── ml_modeler.py         # Moduł do modelowania ML
│   └── utils.py              # Funkcje pomocnicze
│
├── models/                   # Katalog do zapisywania wytrenowanych modeli
│
├── requirements.txt
├── README.md
└── app.py                    # Główna aplikacja Streamlit
```

## Wymagania

- Python 3.8+
- Biblioteki wyszczególnione w pliku `requirements.txt`

## Instalacja

1. Sklonuj repozytorium:
```
git clone https://github.com/user/wine_analysis_app.git
cd wine_analysis_app
```

2. Utwórz i aktywuj wirtualne środowisko (opcjonalnie):
```
python -m venv venv
# W systemie Windows
venv\Scripts\activate
# W systemie Unix/MacOS
source venv/bin/activate
```

3. Zainstaluj wymagane biblioteki:
```
pip install -r requirements.txt
```

4. Przygotuj dane:
   - Pobierz zbiór danych Wine Dataset z UCI Machine Learning Repository
   - Umieść pliki `wine.data`, `wine.names` i `Index` w katalogu `data/`

## Uruchomienie aplikacji

Aby uruchomić aplikację, wykonaj poniższą komendę w głównym katalogu projektu:

```
streamlit run app.py
```

Aplikacja będzie dostępna w przeglądarce pod adresem `http://localhost:8501`.

## Funkcjonalności

### Przegląd danych
- Wyświetlanie podstawowych informacji o zbiorze danych
- Podgląd próbki danych
- Statystyki opisowe
- Rozkład klas

### Analiza statystyczna
- Obliczanie podstawowych miar statystycznych
- Badanie korelacji między cechami
- Identyfikacja wartości odstających
- Testy statystyczne

### Manipulacja danymi
- Selekcja cech i wierszy
- Zastępowanie wartości
- Obsługa brakujących danych
- Skalowanie i standaryzacja
- Kodowanie binarne

### Wizualizacja
- Histogramy
- Wykresy pudełkowe
- Wykresy rozproszenia (2D i 3D)
- Macierze korelacji
- Wykresy par cech
- Wykresy współrzędnych równoległych

### Modelowanie ML
- Klasyfikacja (KNN, SVM, Random Forest)
- Klastrowanie (K-Means, DBSCAN)
- Reguły asocjacyjne (Apriori)
- Wizualizacja wyników
- Wybór i optymalizacja parametrów modeli

## Autorzy

Projekt stworzony jako zadanie uczenia maszynowego, oparty na zbiorze danych Wine Dataset z UCI Machine Learning Repository.