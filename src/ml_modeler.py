"""
Moduł odpowiedzialny za modelowanie uczenia maszynowego.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Importy dla klasyfikacji
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Importy dla klastrowania
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Importy dla reguł asocjacyjnych
from mlxtend.frequent_patterns import apriori, association_rules


class ClassificationModel:
    """
    Klasa do modelowania klasyfikacji dla zbioru danych Wine.
    """

    def __init__(self, model_type: str = 'rf'):
        """
        Inicjalizuje model klasyfikacji.

        Args:
            model_type: Typ modelu ('knn', 'svm', 'rf')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def create_model(self, params: Optional[Dict] = None) -> Any:
        """
        Tworzy model na podstawie podanego typu.

        Args:
            params: Parametry modelu

        Returns:
            Stworzony model
        """
        if params is None:
            params = {}

        # Obsługa różnych wariantów nazw dla K-Means
        if self.model_type.lower().replace('-', '') in ['kmeans', 'kmeans']:
            n_clusters = params.get('n_clusters', 3)
            self.n_clusters = n_clusters
            return KMeans(n_clusters=n_clusters, random_state=42)

        # Obsługa różnych wariantów nazw dla DBSCAN
        elif self.model_type.lower().replace('-', '') in ['dbscan']:
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', 5)
            return DBSCAN(eps=eps, min_samples=min_samples)

        else:
            raise ValueError(f"Nieznany typ modelu: {self.model_type}")

    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Przygotowuje dane do modelowania.

        Args:
            X: Cechy
            y: Zmienna celu
            test_size: Proporcja zbioru testowego
            random_state: Ziarno losowości
        """
        self.feature_names = X.columns.tolist()

        # Podział na zbiór treningowy i testowy
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Normalizacja danych
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names
        )

        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names
        )

    def train(self, params: Optional[Dict] = None) -> Dict:
        """
        Trenuje model klasyfikacji.

        Args:
            params: Parametry modelu

        Returns:
            Słownik z wynikami treningu
        """
        if self.X_train is None or self.y_train is None:
            return {"error": "Dane nie zostały przygotowane. Wywołaj prepare_data() najpierw."}

        # Stwórz model
        self.model = self.create_model(params)

        # Trenuj model
        self.model.fit(self.X_train, self.y_train)

        # Wyniki na zbiorze treningowym
        train_preds = self.model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, train_preds)

        # Wyniki na zbiorze testowym
        test_preds = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_preds)

        # Raport klasyfikacji
        report = classification_report(self.y_test, test_preds, output_dict=True)

        # Macierz pomyłek
        conf_matrix = confusion_matrix(self.y_test, test_preds).tolist()

        # Walidacja krzyżowa
        cv_scores = cross_val_score(self.model, pd.concat([self.X_train, self.X_test]),
                                    pd.concat([self.y_train, self.y_test]),
                                    cv=5, scoring='accuracy')

        results = {
            "model_type": self.model_type,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cross_val_scores": cv_scores.tolist(),
            "cross_val_mean": cv_scores.mean(),
            "cross_val_std": cv_scores.std(),
            "classification_report": report,
            "confusion_matrix": conf_matrix
        }

        # Dodaj ważność cech, jeśli dostępna
        if hasattr(self.model, 'feature_importances_'):
            results["feature_importance"] = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))

        return results

    def optimize_hyperparameters(self, param_grid: Dict) -> Dict:
        """
        Optymalizuje hiperparametry modelu.

        Args:
            param_grid: Siatka parametrów do przeszukania

        Returns:
            Słownik z wynikami optymalizacji
        """
        if self.X_train is None or self.y_train is None:
            return {"error": "Dane nie zostały przygotowane. Wywołaj prepare_data() najpierw."}

        # Stwórz model bazowy
        base_model = self.create_model()

        # Stwórz GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        # Trenuj GridSearchCV
        grid_search.fit(self.X_train, self.y_train)

        # Najlepszy model i jego parametry
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Zapisz najlepszy model
        self.model = best_model

        # Wyniki na zbiorze testowym
        test_preds = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_preds)

        # Wyniki dla wszystkich kombinacji parametrów
        results_df = pd.DataFrame(grid_search.cv_results_)

        # Wybierz tylko najważniejsze kolumny
        results_df = results_df[[
            'params', 'mean_test_score', 'std_test_score',
            'rank_test_score', 'mean_fit_time'
        ]]

        return {
            "best_params": best_params,
            "best_score": grid_search.best_score_,
            "test_accuracy": test_accuracy,
            "all_results": results_df.to_dict(orient='records')
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Wykonuje predykcję na nowych danych.

        Args:
            X: DataFrame z cechami

        Returns:
            Tablica z predykcjami
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany. Wywołaj train() najpierw.")

        # Skalowanie danych
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )

        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Wykonuje predykcję prawdopodobieństw przynależności do klas.

        Args:
            X: DataFrame z cechami

        Returns:
            Tablica z prawdopodobieństwami
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany. Wywołaj train() najpierw.")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model nie obsługuje predykcji prawdopodobieństw.")

        # Skalowanie danych
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )

        return self.model.predict_proba(X_scaled)


class ClusteringModel:
    """
    Klasa do modelowania klastrowania dla zbioru danych Wine.
    """

    def __init__(self, model_type: str = 'kmeans'):
        """
        Inicjalizuje model klastrowania.

        Args:
            model_type: Typ modelu ('kmeans', 'dbscan')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.X = None
        self.feature_names = None
        self.n_clusters = None

    def create_model(self, params: Optional[Dict] = None) -> Any:
        """
        Tworzy model na podstawie podanego typu.

        Args:
            params: Parametry modelu

        Returns:
            Stworzony model
        """
        if params is None:
            params = {}

        # Obsługa różnych wariantów nazw dla K-Means
        model_type_normalized = self.model_type.lower().replace('-', '').replace(' ', '')
        if 'kmeans' in model_type_normalized or 'k-means' in self.model_type.lower():
            n_clusters = params.get('n_clusters', 3)
            self.n_clusters = n_clusters
            return KMeans(n_clusters=n_clusters, random_state=42)

        elif self.model_type == 'dbscan':
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', 5)
            return DBSCAN(eps=eps, min_samples=min_samples)

        else:
            raise ValueError(f"Nieznany typ modelu: {self.model_type}")

    def prepare_data(self, X: pd.DataFrame) -> None:
        """
        Przygotowuje dane do modelowania.

        Args:
            X: Cechy
        """
        self.feature_names = X.columns.tolist()

        # Normalizacja danych
        self.X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_names
        )

    def train(self, params: Optional[Dict] = None) -> Dict:
        """
        Trenuje model klastrowania.

        Args:
            params: Parametry modelu

        Returns:
            Słownik z wynikami klastrowania
        """
        if self.X is None:
            return {"error": "Dane nie zostały przygotowane. Wywołaj prepare_data() najpierw."}

        # Stwórz model
        self.model = self.create_model(params)

        # Trenuj model
        self.model.fit(self.X)

        # Uzyskaj etykiety klastrów
        labels = self.model.labels_

        # Liczba klastrów (dla DBSCAN może być inna niż zdefiniowana)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Liczba próbek w każdym klastrze
        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

        results = {
            "model_type": self.model_type,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes
        }

        # Dodaj dodatkowe informacje specyficzne dla modelu
        if self.model_type == 'kmeans':
            results["inertia"] = self.model.inertia_
            results["cluster_centers"] = self.model.cluster_centers_.tolist()

        # Oblicz silhouette score (jeśli więcej niż jeden klaster)
        if n_clusters > 1 and n_clusters < len(self.X):
            silhouette = silhouette_score(self.X, labels)
            results["silhouette_score"] = silhouette

        return results

    def get_clusters(self) -> pd.Series:
        """
        Zwraca etykiety klastrów.

        Returns:
            Seria z etykietami klastrów
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany. Wywołaj train() najpierw.")

        return pd.Series(self.model.labels_)

    def find_optimal_clusters(self, max_clusters: int = 10) -> Dict:
        """
        Znajduje optymalną liczbę klastrów dla KMeans.

        Args:
            max_clusters: Maksymalna liczba klastrów do sprawdzenia

        Returns:
            Słownik z wynikami
        """
        if self.X is None:
            return {"error": "Dane nie zostały przygotowane. Wywołaj prepare_data() najpierw."}

        if self.model_type != 'kmeans':
            return {"error": "Metoda dostępna tylko dla KMeans."}

        # Listy na wyniki
        inertias = []
        silhouettes = []

        # Przeszukaj różne liczby klastrów
        for k in range(2, max_clusters + 1):
            # Stwórz model KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)

            # Trenuj model
            kmeans.fit(self.X)

            # Zapisz wyniki
            inertias.append(kmeans.inertia_)

            # Oblicz silhouette score
            silhouettes.append(silhouette_score(self.X, kmeans.labels_))

        # Znajdź optymalną liczbę klastrów na podstawie silhouette score
        optimal_k = silhouettes.index(max(silhouettes)) + 2

        # Zapisz wyniki
        results = {
            "inertias": inertias,
            "silhouettes": silhouettes,
            "k_values": list(range(2, max_clusters + 1)),
            "optimal_k": optimal_k
        }

        return results


class AssociationRulesMiner:
    """
    Klasa do wydobywania reguł asocjacyjnych.
    """

    def __init__(self):
        """
        Inicjalizuje model wydobywania reguł asocjacyjnych.
        """
        self.frequent_itemsets = None
        self.rules = None
        self.X = None

    def prepare_data(self, X: pd.DataFrame, threshold: float = 0.5) -> None:
        """
        Przygotowuje dane do wydobywania reguł.

        Args:
            X: DataFrame z cechami
            threshold: Próg dla konwersji wartości na binarne
        """
        # Konwersja wartości na binarne
        self.X = X.copy()

        # Dla każdej kolumny numerycznej
        for col in self.X.select_dtypes(include=np.number).columns:
            # Konwersja na wartości binarne (0 lub 1) na podstawie progu
            self.X[col] = (self.X[col] > self.X[col].mean()).astype(int)

    def find_frequent_itemsets(self, min_support: float = 0.1) -> pd.DataFrame:
        """
        Znajduje częste zbiory elementów.

        Args:
            min_support: Minimalne wsparcie dla zbiorów elementów

        Returns:
            DataFrame z częstymi zbiorami elementów
        """
        if self.X is None:
            raise ValueError("Dane nie zostały przygotowane. Wywołaj prepare_data() najpierw.")

        # Znajdź częste zbiory elementów
        self.frequent_itemsets = apriori(
            self.X,
            min_support=min_support,
            use_colnames=True
        )

        return self.frequent_itemsets

    def generate_rules(self, min_threshold: float = 0.7, metric: str = 'confidence') -> pd.DataFrame:
        """
        Generuje reguły asocjacyjne.

        Args:
            min_threshold: Minimalny próg dla wybranej metryki
            metric: Metryka do oceny reguł ('confidence', 'lift', 'leverage', 'conviction')

        Returns:
            DataFrame z regułami asocjacyjnymi
        """
        if self.frequent_itemsets is None:
            raise ValueError(
                "Częste zbiory elementów nie zostały znalezione. Wywołaj find_frequent_itemsets() najpierw.")

        # Generuj reguły
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )

        return self.rules

    def get_top_rules(self, n: int = 10, metric: str = 'lift') -> pd.DataFrame:
        """
        Zwraca najlepsze reguły według wybranej metryki.

        Args:
            n: Liczba reguł do zwrócenia
            metric: Metryka do sortowania reguł

        Returns:
            DataFrame z najlepszymi regułami
        """
        if self.rules is None:
            raise ValueError("Reguły nie zostały wygenerowane. Wywołaj generate_rules() najpierw.")

        # Sortuj reguły według metryki
        sorted_rules = self.rules.sort_values(metric, ascending=False)

        # Zwróć najlepsze reguły
        return sorted_rules.head(n)

    def format_rules(self, rules: pd.DataFrame) -> List[str]:
        """
        Formatuje reguły do czytelnej postaci.

        Args:
            rules: DataFrame z regułami

        Returns:
            Lista sformatowanych reguł
        """
        formatted_rules = []

        for _, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])

            antecedent_str = ', '.join([f"{col} = 1" for col in antecedents])
            consequent_str = ', '.join([f"{col} = 1" for col in consequents])

            rule_str = f"{antecedent_str} => {consequent_str}"
            rule_str += f" [wsparcie={rule['support']:.3f}, pewność={rule['confidence']:.3f}, lift={rule['lift']:.3f}]"

            formatted_rules.append(rule_str)

        return formatted_rules