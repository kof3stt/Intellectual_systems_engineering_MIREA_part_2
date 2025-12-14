import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from individual import Individual


def goldstein_price(x: float, y: float) -> float:
    """
    Функция Гольдштейна-Прайса для оптимизации.
    Глобальный минимум: f(0, -1) = 3
    """
    term1 = 1 + (x + y + 1) ** 2 * (
        19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2
    )
    term2 = 30 + (2 * x - 3 * y) ** 2 * (
        18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2
    )
    return term1 * term2


class GeneticAlgorithm:
    """Класс генетического алгоритма для оптимизации функции двух переменных"""

    def __init__(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        selection_method: str = "tournament",
        tournament_size: int = 3,
        elitism_count: int = 2,
        crossover_method: str = "arithmetic",
        mutation_method: str = "uniform",
        epsilon=1e8,
    ):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.epsilon = epsilon

        self.best_fitness_history = []

        self.best_individual = None
        self.best_fitness_value = float("inf")

        self.population = self._initialize_population()

    def _initialize_population(self) -> List[Individual]:
        """Инициализация начальной популяции"""
        population = []
        for _ in range(self.population_size):
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            individual = Individual(x, y)
            individual.calculate_fitness(self.fitness_function)
            population.append(individual)
        return population

    def _calculate_fitness_all(self):
        """Вычисление приспособленности для всей популяции"""
        for individual in self.population:
            individual.calculate_fitness(self.fitness_function)

    def _roulette_wheel_selection(self) -> Individual:
        """Селекция методом рулетки для минимизации"""
        max_fitness = max(ind.fitness for ind in self.population)
        inverted_fitness = [
            max_fitness - ind.fitness + 1e-10 for ind in self.population
        ]
        total_inverted = sum(inverted_fitness)

        if total_inverted == 0:
            return random.choice(self.population)

        pick = random.uniform(0, total_inverted)
        current = 0

        for i, individual in enumerate(self.population):
            current += inverted_fitness[i]
            if current >= pick:
                return individual

        return self.population[-1]

    def _tournament_selection(self) -> Individual:
        """Турнирная селекция"""
        tournament = random.sample(self.population, self.tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)

    def _select_parent(self) -> Individual:
        """Выбор родителя в зависимости от метода селекции"""
        if self.selection_method == "roulette":
            return self._roulette_wheel_selection()
        elif self.selection_method == "tournament":
            return self._tournament_selection()
        else:
            return self._tournament_selection()

    def _single_point_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Одноточечное скрещивание"""
        if random.random() > self.crossover_rate:
            return (
                Individual(parent1.x, parent1.y),
                Individual(parent2.x, parent2.y),
            )

        crossover_point = random.randint(0, 1)

        if crossover_point == 0:
            child1 = Individual(parent2.x, parent1.y)
            child2 = Individual(parent1.x, parent2.y)
        else:
            child1 = Individual(parent1.x, parent2.y)
            child2 = Individual(parent2.x, parent1.y)

        return child1, child2

    def _arithmetic_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Арифметическое скрещивание"""
        if random.random() > self.crossover_rate:
            return (
                Individual(parent1.x, parent1.y),
                Individual(parent2.x, parent2.y),
            )

        alpha = random.random()

        x1 = alpha * parent1.x + (1 - alpha) * parent2.x
        x2 = alpha * parent2.x + (1 - alpha) * parent1.x

        y1 = alpha * parent1.y + (1 - alpha) * parent2.y
        y2 = alpha * parent2.y + (1 - alpha) * parent1.y

        child1 = Individual(x1, y1)
        child2 = Individual(x2, y2)

        return child1, child2

    def _uniform_mutation(self, individual: Individual) -> Individual:
        """Равномерная мутация"""
        if random.random() > self.mutation_rate:
            return individual

        if random.random() < 0.5:
            individual.x = random.uniform(self.bounds[0][0], self.bounds[0][1])
        if random.random() < 0.5:
            individual.y = random.uniform(self.bounds[1][0], self.bounds[1][1])

        return individual

    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Скрещивание в зависимости от выбранного метода"""
        if self.crossover_method == "single_point":
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == "arithmetic":
            return self._arithmetic_crossover(parent1, parent2)
        else:
            return self._arithmetic_crossover(parent1, parent2)

    def _apply_elitism(self) -> List[Individual]:
        """Применение элитизма - сохранение лучших особей"""
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness)
        return sorted_population[: self.elitism_count]

    def _update_best_solution(self):
        """Обновление лучшего решения"""
        current_best = min(self.population, key=lambda ind: ind.fitness)
        if current_best.fitness < self.best_fitness_value:
            self.best_fitness_value = current_best.fitness
            self.best_individual = Individual(current_best.x, current_best.y)
            self.best_individual.fitness = current_best.fitness

    def run(self) -> Tuple[Individual, float]:
        """Запуск генетического алгоритма"""
        for generation in range(self.max_generations):

            fitness_values = [ind.fitness for ind in self.population]
            self.best_fitness_history.append(min(fitness_values))

            self._update_best_solution()

            new_population = []

            elites = self._apply_elitism()
            for elite in elites:
                ind = Individual(elite.x, elite.y)
                ind.fitness = elite.fitness
                new_population.append(ind)

            while len(new_population) < self.population_size:
                parent1 = self._select_parent()
                parent2 = self._select_parent()

                child1, child2 = self._crossover(parent1, parent2)

                child1 = self._uniform_mutation(child1)
                child2 = self._uniform_mutation(child2)

                child1.calculate_fitness(self.fitness_function)
                child2.calculate_fitness(self.fitness_function)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

            best_ind = min(self.population, key=lambda ind: ind.fitness)
            actual_value = self.fitness_function(best_ind.x, best_ind.y)
            print(
                f"Поколение {generation}: лучшая приспособленность={best_ind.fitness:.6f}, "
                f"значение функции={actual_value:.6f}, координаты=({best_ind.x:.6f}, {best_ind.y:.6f})"
            )

        self._update_best_solution()

        actual_value = self.fitness_function(
            self.best_individual.x, self.best_individual.y
        )
        print(
            f"Лучшее решение: x={self.best_individual.x:.6f}, y={self.best_individual.y:.6f}"
        )
        print(f"Значение функции: {actual_value:.6f}")

        return self.best_individual, actual_value

    def plot_convergence(self):
        """Построение графика сходимости"""
        plt.figure(figsize=(12, 6))

        generations = list(range(len(self.best_fitness_history)))

        plt.plot(
            generations,
            self.best_fitness_history,
            label="Лучшая приспособленность",
        )

        plt.title("Сходимость генетического алгоритма", fontsize=14)
        plt.xlabel("Поколение", fontsize=12)
        plt.ylabel("Приспособленность", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    bounds = [(-2, 2), (-2, 2)]
    ga = GeneticAlgorithm(
        fitness_function=goldstein_price,
        bounds=bounds,
        population_size=50,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        selection_method="roulette",
        tournament_size=3,
        elitism_count=2,
        crossover_method="arithmetic",
        mutation_method="uniform",
    )

    best_individual, best_value = ga.run()

    ga.plot_convergence()

bounds = [(-2, 2), (-2, 2)]
ga = GeneticAlgorithm(
    fitness_function=goldstein_price,
    bounds=bounds,
    population_size=50,
    max_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    selection_method="tournament",
    tournament_size=3,
    elitism_count=2,
    crossover_method="arithmetic",
    mutation_method="uniform",
)
