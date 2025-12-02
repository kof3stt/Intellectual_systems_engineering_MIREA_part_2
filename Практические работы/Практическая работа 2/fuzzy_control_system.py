import numpy as np
from typing import Dict, List, Tuple
from linguistic_variable import LinguisticVariable
from fuzzy_rules import FuzzyRule, RuleBase


class FuzzyControlSystem:
    """
    Система нечеткого управления с поддержкой:
    - фазификации
    - агрегации правил (Мамдани)
    - аккумуляции
    - дефазификации
    """

    def __init__(self):
        self.input_vars: Dict[str, LinguisticVariable] = {}
        self.output_vars: Dict[str, LinguisticVariable] = {}
        self.rule_base = RuleBase()

    def add_input_variable(self, var: LinguisticVariable):
        self.input_vars[var.name] = var

    def add_output_variable(self, var: LinguisticVariable):
        self.output_vars[var.name] = var

    def add_rule(self, rule: FuzzyRule):
        """Добавляет правило в систему."""
        self.rule_base.add_rule(rule)

    def fuzzify(self, input_values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Фазификация всех входных значений.
        Возвращает словарь: {имя_переменной: {имя_терма: степень_принадлежности}}
        """
        fuzzified = {}
        for var_name, value in input_values.items():
            if var_name in self.input_vars:
                var = self.input_vars[var_name]
                fuzzified[var_name] = {}
                for term_name, term_set in var.terms.items():
                    fuzzified[var_name][term_name] = term_set.mu(value)
        return fuzzified

    def infer_mamdani(
        self,
        input_values: Dict[str, float],
        output_var_name: str,
        num_points: int = 100,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Вывод по методу Мамдани.

        Steps:
        1. Фазификация входов
        2. Оценка правил
        3. Композиция (аккумуляция) выходных термов
        4. Дефазификация
        """
        # 1. Фазификация
        fuzzified_inputs = self.fuzzify(input_values)

        # 2. Оценка правил
        rule_activations = self.rule_base.evaluate_all(fuzzified_inputs)

        # 3. Композиция и аккумуляция
        output_var = self.output_vars[output_var_name]
        universe = np.linspace(output_var.domain_min, output_var.domain_max, num_points)

        aggregated_output = np.zeros_like(universe, dtype=float)

        for (var_name, term_name), activation in rule_activations.items():
            if var_name == output_var_name and activation > 0:
                term_set = output_var.terms[term_name]

                # Получаем функцию принадлежности терма
                membership = np.array([term_set.mu(x) for x in universe])

                # "Обрезаем" функцию по степени активации (метод Мамдани)
                clipped = np.minimum(membership, activation)

                # Аккумулируем с помощью операции максимума
                aggregated_output = np.maximum(aggregated_output, clipped)

        # 4. Дефазификация (центр тяжести)
        if np.sum(aggregated_output) == 0:
            crisp_value = (output_var.domain_min + output_var.domain_max) / 2
        else:
            crisp_value = np.sum(universe * aggregated_output) / np.sum(
                aggregated_output
            )

        return crisp_value, universe, aggregated_output

    def print_system_info(self):
        print("\nВходные переменные:")
        for name, var in self.input_vars.items():
            print(f" {name}: [{var.domain_min:.2f}, {var.domain_max:.2f}]")
            print(f"    Термы: {', '.join(var.get_term_names())}")

        print("\nВыходные переменные:")
        for name, var in self.output_vars.items():
            print(f"  {name}: [{var.domain_min:.2f}, {var.domain_max:.2f}]")
            print(f"    Термы: {', '.join(var.get_term_names())}")

        self.rule_base.print_rules()
