from typing import Dict, List
from fuzzy_sets import FuzzySet
import numpy as np
import matplotlib.pyplot as plt


class LinguisticVariable:
    """
    Лингвистическая переменная: универсум (min,max), дискретизация
    и набор термов (нечетких множеств).
    """

    def __init__(
        self, name: str, domain_min: float, domain_max: float, num_points: int = 100
    ):
        self.name = name
        self.domain_min = float(domain_min)
        self.domain_max = float(domain_max)
        self.num_points = int(num_points)
        self.terms: Dict[str, FuzzySet] = {}

    def add_term(self, fuzzy_set: FuzzySet):
        self.terms[fuzzy_set.name] = fuzzy_set

    def universe(self):
        """Возвращает список дискретных точек универсума."""
        return np.linspace(self.domain_min, self.domain_max, self.num_points)

    def membership_vector(self, term_name: str):
        """Возвращает вектор значений mu(x) по дискретному универсу для заданного терма."""
        if term_name not in self.terms:
            raise KeyError(f"Term {term_name} not found in {self.name}")
        fs = self.terms[term_name]
        xs = self.universe()
        return [fs.mu(x) for x in xs]

    def fuzzify(self, x: float):
        """Возвращает словарь {term: mu(x)} для одной точки x."""
        return {name: fs.mu(x) for name, fs in self.terms.items()}

    def __repr__(self):
        return f"LinguisticVariable({self.name}, [{self.domain_min}, {self.domain_max}], terms={list(self.terms.keys())})"

    def fuzzify_crisp(self, crisp_value: float) -> Dict[str, float]:
        """
        Фазификация четкого значения.
        Возвращает словарь {термин: степень принадлежности} для заданного четкого значения.
        """
        result = {}
        for term_name, fuzzy_set in self.terms.items():
            result[term_name] = fuzzy_set.mu(crisp_value)
        return result

    def plot_terms(self, title: str = None):
        """Визуализация всех терминов лингвистической переменной"""
        x = self.universe()
        plt.figure(figsize=(10, 6))

        for term_name, term_set in self.terms.items():
            y = [term_set.mu(xi) for xi in x]
            plt.plot(x, y, label=term_name, linewidth=2)

        plt.xlabel("Значение")
        plt.ylabel("Степень принадлежности μ")
        plt.title(title or f"Термы лингвистической переменной '{self.name}'")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_term_names(self) -> List[str]:
        """Возвращает список имен терминов."""
        return list(self.terms.keys())

    def get_term(self, term_name: str) -> FuzzySet:
        """Возвращает нечеткое множество для указанного термина."""
        return self.terms[term_name]
