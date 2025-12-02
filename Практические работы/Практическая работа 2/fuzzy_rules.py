from typing import List, Tuple, Dict, Any
import numpy as np


class FuzzyRule:
    """
    Класс для представления одного нечёткого правила.
    Формат: ЕСЛИ (условие1 И условие2 ...) ТО (заключение)
    """

    def __init__(self, name: str = "Неименованное правило"):
        self.name = name
        self.conditions: List[Tuple[str, str, str]] = (
            []
        )  # (var_name, term_name, connective)
        self.conclusion: Tuple[str, str] = None  # (var_name, term_name)

    def add_condition(self, var_name: str, term_name: str, connective: str = "AND"):
        """
        Добавляет условие в правило.

        Args:
            var_name: имя лингвистической переменной
            term_name: имя термина
            connective: "AND"
        """
        self.conditions.append((var_name, term_name, connective))

    def set_conclusion(self, var_name: str, term_name: str):
        """Устанавливает заключение правила."""
        self.conclusion = (var_name, term_name)

    def evaluate(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """
        Вычисляет степень активации правила на основе фазифицированных входов.

        Args:
            fuzzified_inputs: словарь {var_name: {term_name: degree}}

        Returns:
            Степень активации правила (0-1)
        """
        if not self.conditions:
            return 0.0

        activation = 1.0
        for var_name, term_name, connective in self.conditions:
            if connective == "AND":
                if (
                    var_name in fuzzified_inputs
                    and term_name in fuzzified_inputs[var_name]
                ):
                    activation = min(activation, fuzzified_inputs[var_name][term_name])
                else:
                    activation = 0.0

        return activation

    def __str__(self):
        conditions_str = " AND ".join(
            [f"{var} == {term}" for var, term, _ in self.conditions]
        )
        conclusion_str = (
            f"{self.conclusion[0]} == {self.conclusion[1]}"
            if self.conclusion
            else "Нет заключения"
        )
        return f"IF {conditions_str} THEN {conclusion_str}"


class RuleBase:
    """База правил нечёткой системы."""

    def __init__(self):
        self.rules: List[FuzzyRule] = []

    def add_rule(self, rule: FuzzyRule):
        """Добавляет правило в базу."""
        self.rules.append(rule)

    def evaluate_all(
        self, fuzzified_inputs: Dict[str, Dict[str, float]]
    ) -> Dict[Tuple[str, str], float]:
        """
        Оценивает все правила и возвращает степени активации для каждого вывода.

        Returns:
            Словарь {(output_var, term): activation}
        """
        activations = {}

        for rule in self.rules:
            activation = rule.evaluate(fuzzified_inputs)
            if activation > 0 and rule.conclusion:
                key = (rule.conclusion[0], rule.conclusion[1])
                if key in activations:
                    activations[key] = max(activations[key], activation)
                else:
                    activations[key] = activation

        return activations

    def print_rules(self):
        """Выводит все правила."""
        print(f"\nБаза правил ({len(self.rules)} правил):")
        for i, rule in enumerate(self.rules, 1):
            print(f"{i}. {rule}")
