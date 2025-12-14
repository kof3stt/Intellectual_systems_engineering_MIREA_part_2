from typing import Callable


class Individual:
    """Класс, представляющий особь (хромосому) в популяции"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.fitness = 0.0

    def __repr__(self):
        return f"Individual(x={self.x:.6f}, y={self.y:.6f}, fitness={self.fitness:.6f})"

    def calculate_fitness(self, fitness_func: Callable) -> float:
        """Вычисление приспособленности особи"""
        self.fitness = fitness_func(self.x, self.y)
        return self.fitness
