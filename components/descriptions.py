"""
Opisy i informacje pomocy dla różnych sekcji aplikacji.
"""

# Opisy stron
PAGE_DESCRIPTIONS = {
    "overview": {
        "title": "Przegląd danych Wine Dataset",
        "description": """
        Ta sekcja pozwala na zapoznanie się z podstawowymi informacjami o zbiorze danych Wine Dataset.
        Wine Dataset to zbiór danych zawierający wyniki analizy chemicznej win pochodzących z jednego regionu 
        Włoch, ale wytworzonych z trzech różnych odmian winogron. Zawiera 178 próbek z 13 cechami chemicznymi.
        """
    },
    "statistical": {
        "title": "Analiza statystyczna",
        "description": """
        W tej sekcji możesz przeprowadzić szczegółową analizę statystyczną wybranych cech.
        Dostępne są podstawowe statystyki opisowe, analiza rozkładów, korelacji oraz testy statystyczne.
        """
    },
    "manipulation": {
        "title": "Manipulacja danymi",
        "description": """
        Ta sekcja umożliwia manipulację danymi: wybór cech, filtrowanie, zastępowanie wartości,
        obsługę brakujących danych, skalowanie oraz wiele innych operacji na zbiorze danych.
        """
    },
    "visualization": {
        "title": "Wizualizacja danych",
        "description": """
        W tej sekcji możesz tworzyć różnego rodzaju wizualizacje danych, które pomogą lepiej
        zrozumieć strukturę i zależności w zbiorze danych Wine Dataset.
        """
    },
    "ml_modeling": {
        "title": "Modelowanie uczenia maszynowego",
        "description": """
        Ta sekcja umożliwia trenowanie i ewaluację modeli uczenia maszynowego na danych Wine Dataset.
        Możesz wykorzystać algorytmy klasyfikacji, klastrowania oraz reguły asocjacyjne.
        """
    }
}

# Opisy metod statystycznych
STAT_METHODS = {
    "podstawowe_statystyki": {
        "title": "Podstawowe statystyki",
        "description": """
        Podstawowe statystyki opisowe dla wybranych cech:
        - **Minimum**: najmniejsza wartość w danych
        - **Maksimum**: największa wartość w danych
        - **Średnia**: średnia arytmetyczna (suma wartości podzielona przez liczbę obserwacji)
        - **Mediana**: wartość środkowa (rozdziela zbiór na dwie równe części)
        - **Odchylenie standardowe**: miara rozproszenia wartości wokół średniej
        - **Wariancja**: kwadrat odchylenia standardowego
        - **Skośność**: miara asymetrii rozkładu (wartości > 0 oznaczają rozkład z "ogonem" w prawo)
        - **Kurtoza**: miara "ciężkości ogonów" rozkładu (większe wartości oznaczają cięższe ogony)
        """
    },
    "kwartyle": {
        "title": "Kwartyle i percentyle",
        "description": """
        Kwartyle i percentyle dzielą zbiór danych na równe części:
        - **Q25 (1. kwartyl)**: wartość poniżej której znajduje się 25% obserwacji
        - **Q50 (2. kwartyl)**: mediana, wartość poniżej której znajduje się 50% obserwacji
        - **Q75 (3. kwartyl)**: wartość poniżej której znajduje się 75% obserwacji
        - **IQR**: rozstęp międzykwartylowy (Q75 - Q25), miara rozproszenia
        - **P10**: 10. percentyl, wartość poniżej której znajduje się 10% obserwacji
        - **P90**: 90. percentyl, wartość poniżej której znajduje się 90% obserwacji
        """
    },
    "test_normalnosci": {
        "title": "Test normalności (Shapiro-Wilk)",
        "description": """
        Test Shapiro-Wilka bada, czy dane pochodzą z rozkładu normalnego:
        - **Statystyka**: wartość statystyki testowej
        - **P-value**: wartość p; jeśli p < 0.05, hipoteza o normalności rozkładu jest odrzucana
        - **Rozkład normalny**: informacja czy dane można uznać za normalnie rozłożone (przy p > 0.05)

        Rozkład normalny ("dzwonowy") jest istotny dla wielu metod statystycznych i uczenia maszynowego.
        """
    },
    "korelacje": {
        "title": "Analiza korelacji",
        "description": """
        Korelacja mierzy siłę zależności liniowej między dwiema zmiennymi:
        - **Pearson**: mierzy liniową zależność między zmiennymi (-1 do 1)
        - **Spearman**: mierzy monotoniczną zależność między zmiennymi (działa dla danych nienormalnych)
        - **Kendall**: alternatywna miara monotonicznej zależności, odporna na wartości odstające

        Wartości bliskie 1 oznaczają silną dodatnią korelację, wartości bliskie -1 oznaczają
        silną ujemną korelację, a wartości bliskie 0 oznaczają brak korelacji liniowej.
        """
    },
    "wartosci_odstajace": {
        "title": "Analiza wartości odstających",
        "description": """
        Wartości odstające to obserwacje znacznie różniące się od pozostałych:
        - **Metoda IQR**: identyfikuje wartości poza zakresem [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        - **Metoda Z-score**: identyfikuje wartości, których odległość od średniej przekracza 3 odchylenia standardowe

        Wartości odstające mogą wskazywać na błędy pomiaru lub nietypowe przypadki,
        ale mogą też zawierać istotne informacje o badanym zjawisku.
        """
    }
}

# Opisy metod wizualizacji
VISUALIZATION_METHODS = {
    "histogram": {
        "title": "Histogram",
        "description": """
        Histogram pokazuje rozkład wartości zmiennej. Oś X reprezentuje przedziały wartości,
        a oś Y liczbę obserwacji w każdym przedziale. Użyteczny do analizy rozkładu danych,
        identyfikacji skośności, modalności (liczba "szczytów") oraz odstających wartości.
        """
    },
    "boxplot": {
        "title": "Wykres pudełkowy",
        "description": """
        Wykres pudełkowy pokazuje rozkład wartości na podstawie kwartyli. "Pudełko" reprezentuje
        zakres międzykwartylowy (IQR, od Q1 do Q3), linia wewnątrz pudełka to mediana (Q2),
        a "wąsy" rozciągają się do min/max wartości (z wyłączeniem wartości odstających),
        które są zaznaczone jako punkty poza wąsami.
        """
    },
    "scatter": {
        "title": "Wykres rozproszenia",
        "description": """
        Wykres rozproszenia pokazuje zależność między dwiema zmiennymi. Każdy punkt reprezentuje
        jedną obserwację, gdzie pozycja X i Y odpowiada wartościom dwóch wybranych cech. Pozwala
        wykryć wzorce, klastry i korelacje między zmiennymi.
        """
    },
    "scatter3d": {
        "title": "Wykres rozproszenia 3D",
        "description": """
        Wykres rozproszenia 3D rozszerza tradycyjny wykres rozproszenia o trzeci wymiar,
        pozwalając na analizę relacji między trzema zmiennymi jednocześnie. Użyteczny do wykrywania
        złożonych wzorców i klastrów w danych wielowymiarowych.
        """
    },
    "correlation": {
        "title": "Macierz korelacji",
        "description": """
        Macierz korelacji pokazuje siłę zależności (korelacji) między wszystkimi parami zmiennych.
        Kolorowa heatmapa z wartościami od -1 (silna ujemna korelacja) przez 0 (brak korelacji)
        do 1 (silna dodatnia korelacja). Pozwala szybko zidentyfikować najsilniejsze zależności.
        """
    },
    "pairplot": {
        "title": "Wykres par cech",
        "description": """
        Wykres par cech (pairplot) tworzy siatkę wykresów, pokazując relacje między wszystkimi
        parami wybranych zmiennych. Na przekątnej pokazywane są histogramy rozkładów poszczególnych
        zmiennych, a poza przekątną wykresy rozproszenia dla każdej pary zmiennych.
        """
    },
    "parallel": {
        "title": "Współrzędne równoległe",
        "description": """
        Wykres współrzędnych równoległych przedstawia wiele zmiennych jako równoległe osie pionowe.
        Każda obserwacja jest linią łączącą wartości na każdej osi. Pozwala zobaczyć grupy podobnych
        obserwacji i relacje między wieloma zmiennymi jednocześnie.
        """
    }
}

# Opisy metod manipulacji danymi
DATA_MANIPULATION_METHODS = {
    "select_features": {
        "title": "Wybierz cechy",
        "description": """
        Pozwala na wybór konkretnych kolumn (cech) do dalszej analizy. Umożliwia skupienie się
        na najważniejszych cechach i pominięcie tych mniej istotnych lub redundantnych.
        """
    },
    "select_rows": {
        "title": "Wybierz wiersze według klasy",
        "description": """
        Filtruje wiersze wg wartości kolumny Class. Umożliwia analizę wybranego podzbioru danych,
        np. tylko win klasy 1, lub porównanie cech win klasy 1 i 2.
        """
    },
    "replace_values": {
        "title": "Zastąp wartości",
        "description": """
        Zastępuje określone wartości w wybranej kolumnie nowymi wartościami. Może być używane do
        korekty błędów, zastępowania wartości odstających lub tworzenia nowych kategorii.
        """
    },
    "missing_values": {
        "title": "Obsłuż brakujące wartości",
        "description": """
        Wypełnia brakujące wartości (NaN) w danych. Dostępne strategie:
        - **Mean**: zastępuje braki średnią wartością kolumny
        - **Median**: zastępuje braki medianą wartości kolumny
        - **Most frequent**: zastępuje braki najczęściej występującą wartością
        - **Constant**: zastępuje braki stałą wartością

        Obsługa brakujących danych jest kluczowa, ponieważ wiele algorytmów nie działa poprawnie
        z niekompletnymi danymi.
        """
    },
    "duplicates": {
        "title": "Usuń duplikaty",
        "description": """
        Usuwa zduplikowane wiersze z danych. Duplikaty mogą zniekształcać wyniki analizy i modeli,
        zwłaszcza gdy niektóre obserwacje są nadreprezentowane.
        """
    },
    "scaling": {
        "title": "Skaluj dane",
        "description": """
        Transformuje wartości liczbowe do określonego zakresu. Dostępne metody:
        - **Standard**: standaryzacja (z-score), (x - mean) / std, średnia=0, odch.std=1
        - **MinMax**: skalowanie do zakresu [0,1], (x - min) / (max - min)

        Skalowanie jest istotne dla wielu algorytmów ML, które są wrażliwe na skalę cech.
        """
    },
    "encoding": {
        "title": "Kodowanie binarne klasy",
        "description": """
        Przekształca kolumnę Class w zestaw kolumn binarnych (one-hot encoding). Każda kolumna
        reprezentuje jedną klasę (1, 2 lub 3) i zawiera wartości 1 (jeśli obserwacja należy do tej klasy)
        lub 0 (jeśli nie). Ułatwia użycie cech kategorycznych w modelach ML.
        """
    },
    "polynomial": {
        "title": "Dodaj cechy wielomianowe",
        "description": """
        Tworzy nowe cechy jako wielomiany wybranych cech. Na przykład, jeśli mamy cechę X,
        możemy dodać X², X³ itd. Pozwala to modelom liniowym uchwycić nieliniowe zależności.
        """
    }
}

# Opisy modeli uczenia maszynowego
ML_MODELS = {
    "classification": {
        "title": "Klasyfikacja",
        "description": """
        Klasyfikacja to zadanie przypisania obserwacji do jednej z predefiniowanych klas.
        W kontekście Wine Dataset, celem jest przewidzenie klasy wina (1, 2 lub 3)
        na podstawie jego cech chemicznych.

        Dostępne modele klasyfikacyjne:
        - **Random Forest**: Ensemble bazujący na wielu drzewach decyzyjnych, odporny na przeuczenie
        - **K-Nearest Neighbors (KNN)**: Klasyfikuje na podstawie klas k najbliższych sąsiadów
        - **Support Vector Machine (SVM)**: Znajduje hiperpłaszczyznę najlepiej rozdzielającą klasy
        """
    },
    "clustering": {
        "title": "Klastrowanie",
        "description": """
        Klastrowanie to technika nienadzorowanego uczenia maszynowego grupująca podobne obserwacje.
        W przeciwieństwie do klasyfikacji, nie znamy klas z góry - celem jest odkrycie naturalnych
        grup w danych.

        Dostępne metody klastrowania:
        - **K-Means**: Dzieli dane na k grup minimalizując odległości do centroidów
        - **DBSCAN**: Identyfikuje klastry na podstawie gęstości punktów, może wykrywać klastry dowolnych kształtów
        """
    },
    "association_rules": {
        "title": "Reguły asocjacyjne",
        "description": """
        Reguły asocjacyjne to technika odkrywania interesujących relacji między zmiennymi.
        Format reguły: "jeśli X, to Y" (X → Y), gdzie X to poprzednik, a Y to następnik.

        Ważne miary dla reguł asocjacyjnych:
        - **Wsparcie (support)**: Jak często reguła występuje w danych (% obserwacji zawierających X i Y)
        - **Pewność (confidence)**: Jak często Y występuje, gdy wystąpiło X (warunkowe prawdopodobieństwo Y|X)
        - **Lift**: Jak bardziej prawdopodobne jest Y przy wystąpieniu X (stosunek confidence do oczekiwanego)
        """
    }
}

# Opisy parametrów modeli
MODEL_PARAMETERS = {
    "rf": {
        "n_estimators": "Liczba drzew w lesie. Większa liczba zazwyczaj daje lepsze wyniki, ale wydłuża czas treningu.",
        "max_depth": "Maksymalna głębokość drzewa. Ograniczenie głębokości pomaga zapobiegać przeuczeniu.",
        "min_samples_split": "Minimalna liczba próbek wymagana do podziału węzła. Wyższe wartości zapobiegają przeuczeniu."
    },
    "knn": {
        "n_neighbors": "Liczba sąsiadów do uwzględnienia. Mniejsze wartości (np. 1-5) mogą prowadzić do przeuczenia.",
        "weights": "Metoda ważenia głosów: 'uniform' (wszystkie wagi równe) lub 'distance' (wagi zależne od odległości)."
    },
    "svm": {
        "C": "Parametr regularyzacji. Mniejsze wartości oznaczają silniejszą regularyzację (prostszy model).",
        "kernel": "Funkcja jądra: 'linear', 'poly', 'rbf', 'sigmoid'. 'rbf' jest dobrym wyborem ogólnego zastosowania.",
        "gamma": "Współczynnik funkcji jądra. Większe wartości oznaczają mniejszy promień wpływu pojedynczych próbek."
    },
    "kmeans": {
        "n_clusters": "Liczba klastrów do utworzenia. Można znaleźć optymalną wartość metodą łokcia.",
        "init": "Metoda inicjalizacji centroidów: 'k-means++' (inteligentna) lub 'random' (losowa)."
    },
    "dbscan": {
        "eps": "Maksymalna odległość między punktami w tym samym sąsiedztwie. Kluczowy parametr.",
        "min_samples": "Minimalna liczba punktów potrzebna do utworzenia gęstego regionu (klastra)."
    }
}


def get_page_description(page_key):
    """Zwraca opis dla danej strony."""
    if page_key in PAGE_DESCRIPTIONS:
        return PAGE_DESCRIPTIONS[page_key]
    return {"title": "Brak opisu", "description": "Brak opisu dla tej strony."}


def get_stat_method_description(method_key):
    """Zwraca opis dla danej metody statystycznej."""
    if method_key in STAT_METHODS:
        return STAT_METHODS[method_key]
    return {"title": "Brak opisu", "description": "Brak opisu dla tej metody."}


def get_visualization_method_description(method_key):
    """Zwraca opis dla danej metody wizualizacji."""
    if method_key in VISUALIZATION_METHODS:
        return VISUALIZATION_METHODS[method_key]
    return {"title": "Brak opisu", "description": "Brak opisu dla tej metody."}


def get_manipulation_method_description(method_key):
    """Zwraca opis dla danej metody manipulacji danymi."""
    if method_key in DATA_MANIPULATION_METHODS:
        return DATA_MANIPULATION_METHODS[method_key]
    return {"title": "Brak opisu", "description": "Brak opisu dla tej metody."}


def get_ml_model_description(model_key):
    """Zwraca opis dla danego modelu ML."""
    if model_key in ML_MODELS:
        return ML_MODELS[model_key]
    return {"title": "Brak opisu", "description": "Brak opisu dla tego modelu."}


def get_model_parameter_description(model_type, param_key):
    """Zwraca opis dla danego parametru modelu ML."""
    if model_type in MODEL_PARAMETERS and param_key in MODEL_PARAMETERS[model_type]:
        return MODEL_PARAMETERS[model_type][param_key]
    return "Brak opisu dla tego parametru."