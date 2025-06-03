"""
Moduł odpowiedzialny za stronę modelowania uczenia maszynowego w aplikacji Wine Dataset Analysis.
Rozszerzony o opcje wyboru rodzaju eksperymentu zgodnie z wymaganiami prowadzącego.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Import modułów własnych
from src.ml_modeler import ClassificationModel, ClusteringModel, AssociationRulesMiner
from src.data_loader import split_features_target
from src.utils import (
    format_classification_report, format_confusion_matrix,
    generate_model_parameter_description, get_column_types
)
from src.data_visualizer import create_feature_importance
from components.descriptions import (
    get_page_description, get_ml_model_description,
    get_model_parameter_description
)
from components.ui_helpers import show_info_box, section_header


def page_ml_modeling():
    """Wyświetla stronę modelowania uczenia maszynowego."""

    # Pobierz opis strony
    page_info = get_page_description("ml_modeling")

    # Nagłówek strony
    st.header(page_info["title"])
    st.markdown(page_info["description"])

    # Wprowadzenie do uczenia maszynowego
    with st.expander("ℹ️ O uczeniu maszynowym na danych Wine Dataset", expanded=True):
        st.markdown("""
        **Uczenie maszynowe** to technika analizy danych, która umożliwia systemom automatyczne uczenie się i poprawianie
        na podstawie doświadczenia bez jawnego programowania.

        W kontekście zbioru danych Wine Dataset możemy zastosować różne techniki uczenia maszynowego:

        - **Klasyfikacja**: Przewidywanie klasy wina (1, 2 lub 3) na podstawie jego cech chemicznych
        - **Klastrowanie**: Grupowanie podobnych win bez wcześniejszej wiedzy o ich klasach
        - **Reguły asocjacyjne**: Odkrywanie interesujących relacji między cechami chemicznymi win

        Zbiór danych Wine jest idealny do eksperymentowania z uczeniem maszynowym, ponieważ:
        - Jest stosunkowo mały (178 próbek), co umożliwia szybkie trenowanie modeli
        - Zawiera 13 cech, co jest wystarczająco złożone, ale nie przytłaczające
        - Klasy są dość dobrze rozdzielone, co pozwala uzyskać wysoką dokładność

        W tej sekcji możesz trenować i ewaluować różne modele uczenia maszynowego na zbiorze danych Wine.
        """)

    # Wybór kategorii modelu
    st.markdown("---")
    st.subheader("Wybierz typ modelowania")

    model_category = st.selectbox(
        "Kategoria modelu:",
        ["Klasyfikacja", "Klastrowanie", "Reguły asocjacyjne"],
        help="Wybierz typ modelowania uczenia maszynowego do zastosowania"
    )

    # Klasyfikacja
    if model_category == "Klasyfikacja":
        ml_info = get_ml_model_description("classification")
        section_header(ml_info["title"], "Przewidywanie klasy wina na podstawie cech chemicznych")

        with st.expander("ℹ️ O modelach klasyfikacyjnych"):
            st.markdown(ml_info["description"])

        # Sprawdź, czy kolumna Class istnieje
        if 'Class' not in st.session_state.data.columns:
            st.error("Kolumna 'Class' nie istnieje w danych. Nie można wykonać klasyfikacji.")
            return

        # Wybór modelu klasyfikacji
        model_type = st.selectbox(
            "Wybierz model klasyfikacji:",
            ["Random Forest (rf)", "K-Nearest Neighbors (knn)", "Support Vector Machine (svm)"],
            index=0,
            help="Wybierz algorytm klasyfikacji do zastosowania"
        )

        # Mapowanie wyświetlanej nazwy na kod
        model_code = model_type.split(' ')[0].lower()
        if model_code == "random":
            model_code = "rf"
        elif model_code == "k-nearest":
            model_code = "knn"
        elif model_code == "support":
            model_code = "svm"

        # Parametry modelu
        st.subheader("Parametry modelu")

        with st.expander("ℹ️ Co to są parametry modelu?"):
            st.markdown("""
            **Parametry modelu** (hiperparametry) to wartości konfiguracyjne, które kontrolują zachowanie 
            algorytmu uczenia maszynowego. Właściwy dobór parametrów może znacznie poprawić wydajność modelu.

            Każdy model ma inne parametry, które wpływają na różne aspekty jego działania:
            - Jak złożony jest model
            - Jak szybko się uczy
            - Jak dobrze generalizuje na nowe dane

            Dostosowanie tych parametrów do konkretnego problemu i danych nazywa się **strojeniem hiperparametrów**.
            """)

        # Pobierz opisy parametrów dla wybranego modelu
        param_desc = generate_model_parameter_description(model_code)
        params = {}

        # Dynamicznie generuj widgety dla parametrów modelu
        for param_name, param_info in param_desc.items():
            # Dodaj opis parametru
            st.markdown(f"**{param_info['name']}**")
            help_text = get_model_parameter_description(model_code, param_name)

            # Wybierz odpowiedni typ widgetu
            if 'options' in param_info:
                params[param_name] = st.selectbox(
                    f"Wybierz wartość dla {param_info['name']}:",
                    param_info['options'],
                    index=param_info['options'].index(param_info['default']) if param_info['default'] in param_info[
                        'options'] else 0,
                    help=help_text
                )
            elif 'min' in param_info and 'max' in param_info:
                step = param_info.get('step', 1)
                params[param_name] = st.slider(
                    f"Ustaw wartość dla {param_info['name']}:",
                    param_info['min'],
                    param_info['max'],
                    param_info['default'],
                    step=step,
                    help=help_text
                )

        # NOWA SEKCJA: Wybór rodzaju eksperymentu
        st.subheader("🧪 Rodzaj eksperymentu")

        with st.expander("ℹ️ O rodzajach eksperymentów w uczeniu maszynowym", expanded=True):
            st.markdown("""
            **Rodzaj eksperymentu** określa sposób, w jaki model będzie trenowany i ewaluowany:

            - **Train/Test Split**: Dzieli dane na zbiór treningowy i testowy. Model uczony jest na zbiorze treningowym, a ewaluowany na testowym.
            - **Cross-Validation (k-fold)**: Dzieli dane na k części, trenuje model k razy, każdorazowo używając innej części jako zbioru testowego.
            - **Leave-One-Out (LOO)**: Specjalny przypadek cross-validation, gdzie k = liczba próbek. Każda próbka jest raz zbiorem testowym.

            **Zalety poszczególnych metod:**
            - **Train/Test**: Szybki, prosty, dobry dla dużych zbiorów danych
            - **Cross-Validation**: Bardziej niezawodny, lepiej wykorzystuje dane, mniej podatny na przypadkowość podziału
            - **Leave-One-Out**: Maksymalnie wykorzystuje dane, najlepszy dla małych zbiorów, ale czasochłonny
            """)

        # Radio button do wyboru eksperymentu
        experiment_type = st.radio(
            "Wybierz rodzaj eksperymentu:",
            ["Train/Test Split", "Cross-Validation (k-fold)", "Leave-One-Out (LOO)"],
            index=0,
            help="Wybierz metodę podziału danych do treningu i ewaluacji modelu"
        )

        # Parametry eksperymentu w zależności od wyboru
        experiment_params = {}

        if experiment_type == "Train/Test Split":
            col1, col2 = st.columns(2)
            with col1:
                experiment_params['test_size'] = st.slider(
                    "Proporcja zbioru testowego:",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Jaka część danych zostanie użyta do testowania (reszta do treningu)"
                )
            with col2:
                experiment_params['random_state'] = st.number_input(
                    "Ziarno losowości:",
                    min_value=1,
                    max_value=999,
                    value=42,
                    help="Zapewnia powtarzalność wyników"
                )

        elif experiment_type == "Cross-Validation (k-fold)":
            col1, col2 = st.columns(2)
            with col1:
                experiment_params['cv_folds'] = st.slider(
                    "Liczba fałd (k):",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="Na ile części podzielić dane (typowo 5 lub 10)"
                )
            with col2:
                experiment_params['random_state'] = st.number_input(
                    "Ziarno losowości:",
                    min_value=1,
                    max_value=999,
                    value=42,
                    help="Zapewnia powtarzalność wyników",
                    key="cv_random_state"
                )

        else:  # Leave-One-Out
            st.info("Leave-One-Out nie wymaga dodatkowych parametrów. Każda próbka będzie raz zbiorem testowym.")
            experiment_params['cv_folds'] = len(st.session_state.data)  # LOO = n-fold CV gdzie n = liczba próbek

        # Opcja skalowania danych
        st.subheader("Przygotowanie danych")

        scale_data = st.checkbox(
            "Skaluj dane przed trenowaniem",
            value=True,
            help="Zalecane dla większości modeli, szczególnie SVM i KNN"
        )

        # Tworzenie i trening modelu
        if st.button("Trenuj model", key="train_classification"):
            # Przygotowanie danych
            X, y = split_features_target(st.session_state.data)

            # Dodaj pasek postępu
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Trening modelu..."):
                # Aktualizuj pasek postępu
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Trening modelu {model_type}... {i + 1}%")
                    time.sleep(0.01)

                # Inicjalizacja modelu
                model = ClassificationModel(model_code)

                # Wykonaj eksperyment w zależności od wyboru
                if experiment_type == "Train/Test Split":
                    # Przygotowanie danych
                    model.prepare_data(X, y,
                                     test_size=experiment_params['test_size'],
                                     random_state=experiment_params['random_state'])

                    # Trening modelu
                    results = model.train(params)
                    results['experiment_type'] = 'train_test_split'

                elif experiment_type == "Cross-Validation (k-fold)":
                    # Cross-validation
                    results = model.cross_validate(X, y, params,
                                                 cv_folds=experiment_params['cv_folds'],
                                                 random_state=experiment_params['random_state'])
                    results['experiment_type'] = 'cross_validation'

                else:  # Leave-One-Out
                    # Leave-One-Out (LOO = n-fold CV)
                    results = model.cross_validate(X, y, params,
                                                 cv_folds=len(X),  # LOO
                                                 random_state=experiment_params.get('random_state', 42))
                    results['experiment_type'] = 'leave_one_out'

                # Ukryj pasek postępu i tekst statusu po zakończeniu
                progress_bar.empty()
                status_text.empty()

                # Wyświetl komunikat o sukcesie
                st.success(f"Model {model_type} został pomyślnie wytrenowany używając metody: {experiment_type}!")

                # Wyświetlenie wyników w zależności od typu eksperymentu
                display_classification_results(results, experiment_type, model_type)

    # Klastrowanie
    elif model_category == "Klastrowanie":
        ml_info = get_ml_model_description("clustering")
        section_header(ml_info["title"], "Grupowanie win o podobnych cechach chemicznych")

        with st.expander("ℹ️ O modelach klastrowania"):
            st.markdown(ml_info["description"])

        # Wybór modelu klastrowania
        model_type = st.selectbox(
            "Wybierz model klastrowania:",
            ["K-Means", "DBSCAN"],
            index=0,
            help="Wybierz algorytm klastrowania do zastosowania"
        )

        # Mapowanie wyświetlanej nazwy na kod
        model_code = model_type.lower()

        # Parametry modelu
        st.subheader("Parametry modelu")

        # Pobierz opisy parametrów dla wybranego modelu
        param_desc = generate_model_parameter_description(model_code)
        params = {}

        # Dynamicznie generuj widgety dla parametrów modelu
        for param_name, param_info in param_desc.items():
            # Dodaj opis parametru
            st.markdown(f"**{param_info['name']}**")
            help_text = get_model_parameter_description(model_code, param_name)

            # Wybierz odpowiedni typ widgetu
            if 'options' in param_info:
                params[param_name] = st.selectbox(
                    f"Wybierz wartość dla {param_info['name']}:",
                    param_info['options'],
                    index=param_info['options'].index(param_info['default']) if param_info['default'] in param_info[
                        'options'] else 0,
                    help=help_text
                )
            elif 'min' in param_info and 'max' in param_info:
                step = param_info.get('step', 1)
                params[param_name] = st.slider(
                    f"Ustaw wartość dla {param_info['name']}:",
                    param_info['min'],
                    param_info['max'],
                    param_info['default'],
                    step=step,
                    help=help_text
                )

        # Wybór cech do klastrowania
        st.subheader("Wybór cech")

        # Pobierz kolumny numeryczne
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        # Usuń kolumnę Class z listy, jeśli istnieje
        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')

        # Widget do wyboru cech
        selected_features = st.multiselect(
            "Wybierz cechy do klastrowania:",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols,
            help="Wybierz cechy, na podstawie których będą grupowane wina"
        )

        # Opcja porównania z prawdziwymi klasami
        compare_with_true = st.checkbox(
            "Porównaj z prawdziwymi klasami",
            value=True,
            help="Porównaj znalezione klastry z prawdziwymi klasami win (jeśli dostępne)"
        )

        # Przycisk do trenowania modelu
        if st.button("Wykonaj klastrowanie", key="train_clustering") and selected_features:
            # Przygotowanie danych
            X = st.session_state.data[selected_features]

            # Dodaj pasek postępu
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Klastrowanie danych..."):
                # Aktualizuj pasek postępu
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Wykonywanie klastrowania {model_type}... {i + 1}%")
                    time.sleep(0.01)

                # Inicjalizacja modelu
                model = ClusteringModel(model_code)

                # Przygotowanie danych
                model.prepare_data(X)

                # Znajdowanie optymalnej liczby klastrów dla K-Means
                if model_code == 'k-means':
                    st.subheader("Znajdowanie optymalnej liczby klastrów")
                    max_clusters = st.slider(
                        "Maksymalna liczba klastrów do sprawdzenia:",
                        min_value=2,
                        max_value=15,
                        value=10,
                        help="Większa liczba = dłuższy czas obliczeń"
                    )

                    with st.spinner("Szukanie optymalnej liczby klastrów..."):
                        optimal_results = model.find_optimal_clusters(max_clusters)

                        # Wykres metody łokcia (inertia)
                        st.subheader("Metoda łokcia (inertia)")

                        with st.expander("ℹ️ Jak interpretować metodę łokcia?"):
                            st.markdown("""
                            **Metoda łokcia** pomaga znaleźć optymalną liczbę klastrów poprzez wykres inercji
                            (sumy kwadratów odległości punktów od ich centroidów) w zależności od liczby klastrów.

                            Szukamy "łokcia" na wykresie - punktu, w którym dodanie kolejnego klastra daje 
                            znacznie mniejszy spadek inercji. Ten punkt sugeruje optymalną liczbę klastrów.
                            """)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(optimal_results["k_values"], optimal_results["inertias"], 'o-')
                        ax.set_xlabel('Liczba klastrów (k)')
                        ax.set_ylabel('Inertia')
                        ax.set_title('Metoda łokcia dla określenia optymalnej liczby klastrów')
                        ax.grid(True)
                        st.pyplot(fig)

                        # Wykres współczynnika silhouette
                        st.subheader("Współczynnik silhouette")

                        with st.expander("ℹ️ Jak interpretować współczynnik silhouette?"):
                            st.markdown("""
                            **Współczynnik silhouette** mierzy, jak podobny jest obiekt do własnego klastra
                            w porównaniu do innych klastrów. Wartości wahają się od -1 do 1:

                            - **Wartości bliskie 1**: Obiekt jest dobrze przypisany do swojego klastra
                            - **Wartości bliskie 0**: Obiekt jest na granicy między klastrami
                            - **Wartości bliskie -1**: Obiekt prawdopodobnie jest w złym klastrze

                            Wyższe wartości średnie wskazują lepszą konfigurację klastrów.
                            """)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(optimal_results["k_values"], optimal_results["silhouettes"], 'o-')
                        ax.set_xlabel('Liczba klastrów (k)')
                        ax.set_ylabel('Współczynnik silhouette')
                        ax.set_title('Współczynnik silhouette dla określenia optymalnej liczby klastrów')
                        ax.grid(True)
                        st.pyplot(fig)

                        # Informacja o optymalnej liczbie klastrów
                        st.info(
                            f"Optymalna liczba klastrów na podstawie współczynnika silhouette: {optimal_results['optimal_k']}")

                        # Aktualizacja liczby klastrów
                        params['n_clusters'] = optimal_results['optimal_k']

                # Trening modelu
                results = model.train(params)

                # Ukryj pasek postępu i tekst statusu po zakończeniu
                progress_bar.empty()
                status_text.empty()

                # Wyświetl komunikat o sukcesie
                st.success(f"Klastrowanie {model_type} zostało pomyślnie wykonane!")

                # Dodaj etykiety klastrów do danych
                cluster_labels = model.get_clusters()
                clustering_result = X.copy()
                clustering_result['Klaster'] = cluster_labels

                # Jeśli istnieje kolumna Class, dodaj ją do wyników
                if 'Class' in st.session_state.data.columns and compare_with_true:
                    clustering_result['Prawdziwa_klasa'] = st.session_state.data['Class']

                # Wyświetlenie wyników
                st.subheader("Wyniki klastrowania")

                # Podstawowe informacje
                n_clusters = results['n_clusters']
                st.write(f"Liczba klastrów: {n_clusters}")

                # Liczba próbek w każdym klastrze
                st.subheader("Rozkład klastrów")
                cluster_sizes = pd.DataFrame.from_dict(
                    results['cluster_sizes'],
                    orient='index',
                    columns=['Liczba próbek']
                )
                cluster_sizes.index.name = 'Klaster'
                st.dataframe(cluster_sizes)

                # Wykres rozkładu klastrów
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(cluster_sizes.index.astype(str), cluster_sizes['Liczba próbek'])
                ax.set_xlabel('Klaster')
                ax.set_ylabel('Liczba próbek')
                ax.set_title('Rozkład próbek w klastrach')
                ax.grid(axis='y')
                st.pyplot(fig)

                # Dodatkowe informacje dla K-Means
                if model_code == 'k-means':
                    st.subheader("Inertia")
                    st.write(f"Inertia: {results['inertia']:.3f}")

                    with st.expander("ℹ️ Co to jest inertia?"):
                        st.markdown("""
                        **Inertia** to suma kwadratów odległości każdej próbki od centroidu jej klastra.

                        Mniejsza wartość inertia oznacza, że punkty są bliżej swoich centroidów,
                        co sugeruje lepszy podział klastrów.
                        """)

                # Współczynnik silhouette (jeśli dostępny)
                if 'silhouette_score' in results:
                    st.subheader("Współczynnik silhouette")

                    score = results['silhouette_score']
                    st.write(f"Współczynnik silhouette: {score:.3f}")

                    # Interpretacja wyniku
                    if score > 0.7:
                        st.success("Silna struktura klastrów.")
                    elif score > 0.5:
                        st.info("Średnia struktura klastrów.")
                    elif score > 0.25:
                        st.warning("Słaba struktura klastrów.")
                    else:
                        st.error("Brak znaczącej struktury klastrów.")

                # Porównanie z prawdziwymi klasami (jeśli dostępne)
                if 'Class' in st.session_state.data.columns and compare_with_true:
                    st.subheader("Porównanie z prawdziwymi klasami")

                    # Tabela pokazująca liczbę win z każdej klasy w każdym klastrze
                    cross_tab = pd.crosstab(
                        clustering_result['Klaster'],
                        clustering_result['Prawdziwa_klasa'],
                        rownames=['Klaster'],
                        colnames=['Prawdziwa klasa']
                    )

                    st.dataframe(cross_tab)

                    # Wizualizacja porównania
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cross_tab.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_xlabel('Klaster')
                    ax.set_ylabel('Liczba próbek')
                    ax.set_title('Rozkład prawdziwych klas w klastrach')
                    ax.legend(title='Prawdziwa klasa')
                    st.pyplot(fig)

                    # Obliczanie czystości klastrów
                    cluster_purity = np.sum([np.max(cross_tab.values[i]) for i in range(len(cross_tab))]) / np.sum(
                        cross_tab.values)
                    st.metric(
                        "Czystość klastrów",
                        f"{cluster_purity:.3f}",
                        help="Procent próbek w każdym klastrze należących do klasy najczęściej występującej w tym klastrze"
                    )

                    with st.expander("ℹ️ Co to jest czystość klastrów?"):
                        st.markdown("""
                        **Czystość klastrów** (cluster purity) to miara, która pokazuje, jak dobrze klastry odpowiadają prawdziwym klasom.

                        Dla każdego klastra znajdujemy klasę, która występuje w nim najczęściej. Następnie sumujemy liczbę próbek 
                        należących do tych klas dominujących i dzielimy przez całkowitą liczbę próbek.

                        Wartość 1.0 oznacza, że każdy klaster zawiera próbki tylko z jednej klasy.
                        Niższe wartości oznaczają, że klastry zawierają mieszankę próbek z różnych klas.
                        """)

                # Wizualizacja klastrów na wykresie 2D
                if len(selected_features) >= 2:
                    st.subheader("Wizualizacja klastrów (2D)")

                    # Wybór cech do wizualizacji
                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        feat1 = st.selectbox(
                            "Pierwsza cecha (oś X):",
                            selected_features,
                            index=0,
                            help="Wybierz cechę do wyświetlenia na osi X"
                        )

                    with viz_col2:
                        other_feats = [f for f in selected_features if f != feat1]
                        feat2 = st.selectbox(
                            "Druga cecha (oś Y):",
                            other_feats,
                            index=0 if len(other_feats) > 0 else None,
                            help="Wybierz cechę do wyświetlenia na osi Y"
                        )

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
                    if model_code == 'k-means' and 'cluster_centers' in results:
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
                    ax.grid(True)

                    # Dodaj legendę z etykietami klastrów
                    legend = ax.legend(*scatter.legend_elements(), title="Klastry")
                    ax.add_artist(legend)

                    st.pyplot(fig)

                    # Jeśli dostępne są prawdziwe klasy, pokaż drugi wykres porównawczy
                    if 'Class' in st.session_state.data.columns and compare_with_true:
                        fig, ax = plt.subplots(figsize=(10, 8))

                        scatter = ax.scatter(
                            X[feat1],
                            X[feat2],
                            c=st.session_state.data['Class'],
                            cmap='plasma',
                            alpha=0.8,
                            s=50
                        )

                        ax.set_xlabel(feat1)
                        ax.set_ylabel(feat2)
                        ax.set_title(f'Prawdziwe klasy: {feat1} vs {feat2}')
                        ax.grid(True)

                        # Dodaj legendę z etykietami klas
                        legend = ax.legend(*scatter.legend_elements(), title="Prawdziwe klasy")
                        ax.add_artist(legend)

                        st.pyplot(fig)

                # Podsumowanie wyników
                st.subheader("Podsumowanie klastrowania")
                show_info_box("Interpretacja wyników", f"""
                **Model {model_type}** znalazł {n_clusters} klastrów w danych.

                **Kluczowe wnioski:**
                1. {"Klastry mają podobne rozmiary, co sugeruje zrównoważoną strukturę danych." if np.std(list(results['cluster_sizes'].values())) / np.mean(list(results['cluster_sizes'].values())) < 0.3 else "Klastry mają różne rozmiary, co może wskazywać na naturalne grupowanie danych lub szum."}
                2. {"Wysoki współczynnik silhouette sugeruje dobrze odseparowane klastry." if 'silhouette_score' in results and results['silhouette_score'] > 0.5 else "Umiarkowany współczynnik silhouette sugeruje, że klastry częściowo się nakładają." if 'silhouette_score' in results else ""}

                **Zalecenia:**
                - {"Spróbuj zmniejszyć liczbę klastrów, jeśli chcesz bardziej ogólny podział." if n_clusters > 5 else "Spróbuj zwiększyć liczbę klastrów, jeśli chcesz bardziej szczegółowy podział." if n_clusters < 3 else "Liczba klastrów wydaje się odpowiednia dla tych danych."}
                """)

        elif not selected_features and st.button("Wykonaj klastrowanie", key="no_features"):
            st.error("Wybierz co najmniej jedną cechę do klastrowania.")

    # Reguły asocjacyjne
    elif model_category == "Reguły asocjacyjne":
        ml_info = get_ml_model_description("association_rules")
        section_header(ml_info["title"], "Odkrywanie interesujących relacji między cechami chemicznymi win")

        with st.expander("ℹ️ O regułach asocjacyjnych"):
            st.markdown(ml_info["description"])

        # Wybór cech
        st.subheader("Wybór cech")

        # Pobierz kolumny numeryczne
        column_types = get_column_types(st.session_state.data)
        numeric_cols = column_types.get('numeric', [])

        # Widget do wyboru cech
        selected_features = st.multiselect(
            "Wybierz cechy do analizy:",
            numeric_cols,
            default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols,
            help="Wybierz cechy, między którymi chcesz znaleźć relacje"
        )

        # Parametry
        st.subheader("Parametry")

        # Próg binaryzacji
        threshold = st.slider(
            "Próg binaryzacji (percentyl):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Wartości powyżej tego percentyla będą traktowane jako 1, poniżej jako 0"
        )

        # Wsparcie i pewność
        col1, col2 = st.columns(2)

        with col1:
            min_support = st.slider(
                "Minimalne wsparcie:",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Minimalny procent obserwacji zawierających zbiór elementów"
            )

        with col2:
            min_confidence = st.slider(
                "Minimalna pewność:",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimalna pewność reguły (wsparcie(X,Y) / wsparcie(X))"
            )

        # Lift i metryka
        col3, col4 = st.columns(2)

        with col3:
            min_lift = st.slider(
                "Minimalny lift:",
                min_value=1.0,
                max_value=5.0,
                value=1.2,
                step=0.1,
                help="Minimalna wartość liftu (wsparcie(X,Y) / (wsparcie(X) * wsparcie(Y)))"
            )

        with col4:
            metric = st.selectbox(
                "Metryka do sortowania reguł:",
                ["confidence", "lift", "leverage", "conviction"],
                index=1,
                help="""
                - Confidence: pewność reguły
                - Lift: jak bardziej prawdopodobne jest Y przy wystąpieniu X
                - Leverage: różnica między obserwowanym a oczekiwanym wsparciem
                - Conviction: jak wiele razy reguła byłaby nieprawdziwa, gdyby X i Y były niezależne
                """
            )

        # Tworzenie i trenowanie modelu
        if st.button("Znajdź reguły asocjacyjne", key="train_association_rules") and selected_features:
            # Przygotowanie danych
            X = st.session_state.data[selected_features]

            # Dodaj pasek postępu
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Wydobywanie reguł..."):
                # Aktualizuj pasek postępu
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Wydobywanie reguł asocjacyjnych... {i + 1}%")
                    time.sleep(0.01)

                # Inicjalizacja modelu
                miner = AssociationRulesMiner()

                # Przygotowanie danych
                miner.prepare_data(X, threshold)

                # Znajdź częste zbiory elementów
                frequent_itemsets = miner.find_frequent_itemsets(min_support)

                # Generuj reguły
                min_threshold = min_confidence if metric == 'confidence' else min_lift
                rules = miner.generate_rules(min_threshold, metric)

                # Ukryj pasek postępu i tekst statusu po zakończeniu
                progress_bar.empty()
                status_text.empty()

                # Wyświetl wyniki
                st.subheader("Częste zbiory elementów")

                with st.expander("ℹ️ Co to są częste zbiory elementów?"):
                    st.markdown("""
                    **Częste zbiory elementów** to grupy elementów (cech), które często występują razem.

                    W kontekście Wine Dataset, element może oznaczać "cecha X ma wysoką wartość".
                    Na przykład, zbiór {Alcohol, Flavanoids} oznacza, że wina często mają jednocześnie
                    wysoką zawartość alkoholu i flawonoidów.

                    Kolumna **support** pokazuje, jaki procent wszystkich win ma jednocześnie wszystkie
                    elementy w danym zbiorze.
                    """)

                if frequent_itemsets is not None and not frequent_itemsets.empty:
                    st.write(f"Znaleziono {len(frequent_itemsets)} częstych zbiorów elementów.")

                    # Pokaż najczęstsze zbiory
                    st.dataframe(frequent_itemsets.sort_values('support', ascending=False).head(20))
                else:
                    st.info("Nie znaleziono częstych zbiorów elementów z podanymi parametrami.")

                st.subheader("Reguły asocjacyjne")

                with st.expander("ℹ️ Jak interpretować reguły asocjacyjne?"):
                    st.markdown("""
                    **Reguły asocjacyjne** mają format "jeśli X, to Y" (X → Y):

                    - **X** (antecedents) to poprzednik/warunek
                    - **Y** (consequents) to następnik/konsekwencja

                    Przykład: {wysokie_Alcohol, wysokie_Flavanoids} → {wysokie_Proline}

                    Miary jakości reguł:
                    - **Support**: jaki procent wszystkich próbek zawiera zarówno X, jak i Y
                    - **Confidence**: jaki procent próbek zawierających X zawiera również Y
                    - **Lift**: jak wiele razy bardziej prawdopodobne jest Y, gdy występuje X

                    Reguły z wyższymi wartościami tych miar są bardziej interesujące.
                    """)

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
                    scatter = ax.scatter(
                        viz_rules['support'],
                        viz_rules['confidence'],
                        s=viz_rules['lift'] * 100,
                        alpha=0.6
                    )

                    # Dodaj etykiety dla punktów
                    for i, label in enumerate(rule_labels):
                        ax.annotate(
                            label,
                            (viz_rules['support'].iloc[i], viz_rules['confidence'].iloc[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=8
                        )

                    ax.set_xlabel('Wsparcie (support)')
                    ax.set_ylabel('Pewność (confidence)')
                    ax.set_title('Reguły asocjacyjne: wsparcie vs pewność (rozmiar = lift)')
                    ax.grid(True)

                    # Dodaj kolorową legendę dla liftu
                    handles, labels = scatter.legend_elements(
                        prop="sizes",
                        alpha=0.6,
                        func=lambda s: s / 100
                    )
                    ax.legend(handles, labels, title="Lift", loc="upper left")

                    st.pyplot(fig)

                    # Interpretacja wyników
                    st.subheader("Interpretacja wyników")
                    show_info_box("Najważniejsze wnioski", f"""
                    Z analizy reguł asocjacyjnych można wyciągnąć następujące wnioski:

                    1. **Najsilniejsze powiązania** występują między: {', '.join([f"{','.join(list(r['antecedents']))} => {','.join(list(r['consequents']))}" for _, r in top_rules.head(3).iterrows()])}

                    2. **Najczęstsze cechy** występujące w regułach: {', '.join(set([item for _, row in top_rules.iterrows() for itemset in [row['antecedents'], row['consequents']] for item in itemset]))}

                    3. **Praktyczne zastosowanie**: Reguły o wysokiej pewności i lifcie mogą być używane do przewidywania cech chemicznych win na podstawie innych cech.

                    4. **Interesujące wzorce**: Wina z wysoką zawartością jednych związków często mają również wysoką zawartość innych związków, co może odzwierciedlać procesy biochemiczne zachodzące podczas produkcji wina.
                    """)
                else:
                    st.info("Nie znaleziono reguł asocjacyjnych z podanymi parametrami.")

                    # Sugestie
                    st.warning("""
                    **Sugestie:**
                    - Spróbuj zmniejszyć wartość minimalnego wsparcia
                    - Spróbuj zmniejszyć wartość minimalnej pewności lub liftu
                    - Wybierz więcej cech do analizy
                    """)

        elif not selected_features and st.button("Znajdź reguły asocjacyjne", key="no_features_rules"):
            st.error("Wybierz co najmniej jedną cechę do analizy.")

    # Podsumowanie
    st.markdown("---")
    st.subheader("Porównanie metod uczenia maszynowego")

    with st.expander("📚 Wskazówki wyboru odpowiedniej metody"):
        st.markdown("""
        ### Kiedy używać różnych metod uczenia maszynowego:

        **Klasyfikacja:**
        - Gdy masz oznaczone dane (wiesz, do której klasy należy każda próbka)
        - Gdy celem jest przewidywanie kategorii dla nowych próbek
        - Gdy chcesz zrozumieć, które cechy są najważniejsze dla rozróżnienia klas

        **Klastrowanie:**
        - Gdy nie masz oznaczonych danych lub chcesz odkryć naturalne grupowania
        - Gdy chcesz znaleźć ukryte struktury lub segmenty w danych
        - Gdy chcesz zredukować złożoność danych, grupując podobne próbki

        **Reguły asocjacyjne:**
        - Gdy chcesz odkryć interesujące relacje między cechami
        - Gdy szukasz wzorców współwystępowania cech
        - Gdy chcesz generować reguły, które można łatwo interpretować

        ### Wybór rodzaju eksperymentu dla klasyfikacji:

        **Train/Test Split:**
        - Szybki i prosty
        - Dobry dla dużych zbiorów danych
        - Może być mniej stabilny dla małych zbiorów

        **Cross-Validation:**
        - Bardziej niezawodny niż train/test split
        - Lepiej wykorzystuje dostępne dane
        - Daje bardziej stabilne oszacowanie wydajności

        **Leave-One-Out:**
        - Maksymalnie wykorzystuje dane
        - Najlepszy dla bardzo małych zbiorów danych
        - Może być czasochłonny dla dużych zbiorów
        - Daje najlepsze oszacowanie wydajności dla małych zbiorów
        """)

    # Końcowe podsumowanie
    show_info_box("Podsumowanie modelowania uczenia maszynowego", """
    **Najlepsze praktyki modelowania:**

    1. **Wybierz odpowiedni eksperyment**: Dla małych zbiorów jak Wine Dataset, Cross-Validation lub LOO dają bardziej wiarygodne wyniki
    2. **Przygotuj dane odpowiednio**: Skaluj cechy dla większości modeli, szczególnie SVM i KNN
    3. **Wybierz odpowiednie cechy**: Nie wszystkie cechy są równie ważne, czasem mniej cech daje lepsze wyniki
    4. **Dostosuj parametry**: Każdy model ma parametry, które można dostosować do konkretnego problemu
    5. **Waliduj model**: Zawsze oceniaj model na danych, których nie widział podczas treningu
    6. **Interpretuj wyniki**: Sama dokładność to nie wszystko, ważne jest zrozumienie, dlaczego model podejmuje takie decyzje

    Najważniejszym elementem jest dopasowanie algorytmu i metody ewaluacji do problemu i danych, które masz.
    """)


def display_classification_results(results, experiment_type, model_type):
    """
    Wyświetla wyniki klasyfikacji w zależności od typu eksperymentu.

    Args:
        results: Słownik z wynikami
        experiment_type: Typ eksperymentu
        model_type: Typ modelu
    """

    # Nagłówek wyników
    st.subheader(f"Wyniki klasyfikacji - {experiment_type}")

    if experiment_type == "Train/Test Split":
        # Tradycyjne wyniki train/test split
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Dokładność (zbiór treningowy)",
                f"{results['train_accuracy']:.3f}",
                help="Procent poprawnych przewidywań na danych treningowych"
            )

        with col2:
            st.metric(
                "Dokładność (zbiór testowy)",
                f"{results['test_accuracy']:.3f}",
                help="Procent poprawnych przewidywań na danych testowych"
            )

        with col3:
            st.metric(
                "Dokładność (walidacja krzyżowa)",
                f"{results['cross_val_mean']:.3f} ± {results['cross_val_std']:.3f}",
                help="Średnia dokładność z 5-krotnej walidacji krzyżowej ± odchylenie standardowe"
            )

        # Raport klasyfikacji
        st.subheader("Raport klasyfikacji")
        report_df = format_classification_report(results['classification_report'])
        st.dataframe(report_df)

        # Macierz pomyłek
        st.subheader("Macierz pomyłek")
        y_test = results.get('y_test', [])
        if len(y_test) > 0:
            classes = sorted(set(y_test))
            conf_matrix_df = format_confusion_matrix(results['confusion_matrix'], [str(c) for c in classes])
            st.dataframe(conf_matrix_df)

    else:  # Cross-Validation lub Leave-One-Out
        # Wyniki cross-validation
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Średnia dokładność",
                f"{results['cv_mean']:.3f}",
                help="Średnia dokładność ze wszystkich fałd"
            )

        with col2:
            st.metric(
                "Odchylenie standardowe",
                f"{results['cv_std']:.3f}",
                help="Odchylenie standardowe dokładności między fałdami"
            )

        with col3:
            confidence_interval = f"[{results['cv_mean'] - 1.96*results['cv_std']:.3f}, {results['cv_mean'] + 1.96*results['cv_std']:.3f}]"
            st.metric(
                "95% przedział ufności",
                confidence_interval,
                help="95% przedział ufności dla dokładności"
            )

        # Wyniki dla poszczególnych fałd
        if 'cv_scores' in results:
            st.subheader("Wyniki dla poszczególnych fałd")

            scores_df = pd.DataFrame({
                'Fałda': range(1, len(results['cv_scores']) + 1),
                'Dokładność': results['cv_scores']
            })

            st.dataframe(scores_df, use_container_width=True)

            # Wykres dokładności dla poszczególnych fałd
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(scores_df['Fałda'], scores_df['Dokładność'], 'o-', linewidth=2, markersize=8)
            ax.axhline(y=results['cv_mean'], color='r', linestyle='--', alpha=0.7, label=f'Średnia: {results["cv_mean"]:.3f}')
            ax.fill_between(scores_df['Fałda'],
                           results['cv_mean'] - results['cv_std'],
                           results['cv_mean'] + results['cv_std'],
                           alpha=0.2, color='red', label=f'±1 SD')
            ax.set_xlabel('Numer fałdy')
            ax.set_ylabel('Dokładność')
            ax.set_title(f'Dokładność modelu dla poszczególnych fałd ({experiment_type})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # Ważność cech (dla Random Forest)
    if 'feature_importance' in results:
        st.subheader("Ważność cech")

        # Sortuj cechy według ważności
        feature_imp = results['feature_importance']
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)

        features = [x[0] for x in sorted_features]
        importance = [x[1] for x in sorted_features]

        # Stwórz wykres ważności cech
        fig = create_feature_importance(features, importance)
        st.pyplot(fig)

        # Wyświetl tabelę z ważnością cech
        importance_df = pd.DataFrame({
            'Cecha': features,
            'Ważność': importance
        }).sort_values('Ważność', ascending=False)

        st.dataframe(importance_df)