import matplotlib.pyplot as plt
from math import exp
from typing import Dict, Union, Callable, Optional


class FuzzySet:
    """
    Универсальный класс нечеткого множества.
    """
    
    def __init__(self, name: str, data: Union[Dict, Callable] = None):
        """
        Args:
            name: имя множества
            data: либо словарь {элемент: степень_принадлежности}, 
                  либо функция принадлежности mu(x)
        """
        self.name = name
        if isinstance(data, dict):
            self.data_type = "discrete"
            self.data = data
        elif callable(data):
            self.data_type = "continuous"
            self.mu_func = data
        else:
            self.data_type = "empty"
            self.data = {}
    
    def mu(self, x):
        """Возвращает степень принадлежности x к множеству в [0,1]."""
        if self.data_type == "discrete":
            return self.data.get(x, 0.0)
        elif self.data_type == "continuous":
            val = self.mu_func(x)
            return max(0.0, min(1.0, val))
        return 0.0
    
    def __call__(self, x):
        return self.mu(x)
    
    def get_elements(self):
        """Возвращает список элементов (для дискретного множества)."""
        if self.data_type == "discrete":
            return list(self.data.keys())
        return []
    
    def get_values(self):
        """Возвращает список значений принадлежности (для дискретного множества)."""
        if self.data_type == "discrete":
            return list(self.data.values())
        return []
    
    def plot(self):
        """Визуализация дискретного множества."""
        if self.data_type != "discrete":
            print(f"Множество {self.name} не является дискретным для визуализации")
            return
        
        elements = self.get_elements()
        values = self.get_values()
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(elements)), values)
        plt.xticks(range(len(elements)), elements, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.ylabel('μ')
        plt.title(self.name)
        plt.tight_layout()
        plt.show()

    def apply_modus_ponens(self, relation) -> Optional['FuzzySet']:
        """
        Применяет правило modus ponens: B' = A' ∘ (A → B)
        
        Args:
            relation: отношение A → B
        Returns:
            Нечёткое множество B'
        """
        if self.data_type != "discrete" or relation.rows != self.get_elements():
            print("Невозможно применить modus ponens: несовместимые множества")
            return None
        
        b_prime = {}
        elements_a = self.get_elements()
        elements_b = relation.columns
        
        for j, elem_b in enumerate(elements_b):
            max_val = 0.0
            for i, elem_a in enumerate(elements_a):
                mu_a = self.mu(elem_a)
                mu_r = relation.matrix[i][j]
                max_val = max(max_val, min(mu_a, mu_r))
            b_prime[elem_b] = round(max_val, 2)
        
        return FuzzySet(f"Вывод из {self.name}", b_prime)


    def complement(self) -> 'FuzzySet':
        """Возвращает дополнение нечёткого множества."""
        if self.data_type == "discrete":
            comp_data = {k: 1.0 - v for k, v in self.data.items()}
            return FuzzySet(f"Не {self.name}", comp_data)
        else:
            def comp_func(x):
                return 1.0 - self.mu(x)
            return FuzzySet(f"Не {self.name}", comp_func)
