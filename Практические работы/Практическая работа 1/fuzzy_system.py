from typing import Dict, List
import numpy as np
from linguistic_variable import LinguisticVariable
from relations import FuzzyRelation


class FuzzySystem:
    """
    Класс для хранения переменных, правил (отношений) и выполнения операций:
    - построение отношения
    - композиция отношений
    - применение правила (вывод)
    """

    def __init__(self):
        self.variables: Dict[str, LinguisticVariable] = {}
        self.relations: Dict[str, FuzzyRelation] = {}

    def add_variable(self, var: LinguisticVariable):
        self.variables[var.name] = var

    def build_relation(
        self,
        var_x_name: str,
        term_x: str,
        var_y_name: str,
        term_y: str,
        relation_name: str,
    ):
        """
        Построить отношение R = term_x * term_y (product via min),
        дискретизация берется из переменных.
        """
        X = self.variables[var_x_name]
        Y = self.variables[var_y_name]
        x_uni = X.universe()
        y_uni = Y.universe()
        mu_x = X.membership_vector(term_x)
        mu_y = Y.membership_vector(term_y)
        R = FuzzyRelation.from_product(x_uni, y_uni, mu_x, mu_y)
        self.relations[relation_name] = R
        return R

    def compose_relations(self, r_name: str, s_name: str, out_name: str):
        R = self.relations[r_name]
        S = self.relations[s_name]
        T = R.compose_max_min(S)
        self.relations[out_name] = T
        return T

    def apply_rule(self, a_values: List[float], relation_name: str) -> List[float]:
        """
        Применение правила: A' ○ R = B' , где A' — вектор степеней по дискретному универсуму X.
        Вход a_values должен иметь длину len(R.x_universe).
        Возвращает вектор длины len(R.y_universe) — B'.
        """
        R = self.relations[relation_name]
        a = np.array(a_values, dtype=float)
        assert a.shape[0] == R.n
        b = np.zeros(R.m, dtype=float)
        for j in range(R.m):
            mins = [min(a[i], float(R.matrix[i, j])) for i in range(R.n)]
            b[j] = max(mins) if mins else 0.0
        return b.tolist()
