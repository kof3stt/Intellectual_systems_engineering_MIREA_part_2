from typing import Dict
from fuzzy_sets import FuzzySet
import numpy as np


class LinguisticVariable:
    """
    Лингвистическая переменная: универсум (min,max), дискретизация
    и набор терминов (нечетких множеств).
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
