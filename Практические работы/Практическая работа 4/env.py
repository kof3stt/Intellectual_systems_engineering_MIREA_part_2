import random
import numpy as np
import pygame


GRID_WIDTH = 10
GRID_HEIGHT = 10
BLOCK_SIZE = 50 # Размер одной клетки в пикселях
FPS = 60

# Направления движения
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class SnakeGame:
    def __init__(self, render_mode=None):
        self.render_mode = render_mode # Режим отрисовки (None для обучения без графики и "human" для визуализации)
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE)
            )
            self.clock = pygame.time.Clock() # Объект для контроля FPS
            self.font = pygame.font.Font(None, 36) # Шрифт для отображения счёта

        self.reset() # Инициализация начального состояния среды

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)] # Змейка представлена списком координат сегментов, первый элемент — голова
        self.direction = RIGHT # Начальное направление движения
        self.food = self._generate_food() # Создание еды в случайной позиции
        self.score = 0
        self.done = False # терминальное состояние эпизода
        self.steps_without_food = 0 # контроль за зацикливанием
        return self.get_state()

    def _generate_food(self):
        while True: # Генерируем до тех пор, пока еда не окажется не на змейке.
            food = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1),
            )
            if food not in self.snake:
                return food

    def _danger_ahead(self):
        '''
        Проверка опасности впереди
        
        Возвращает:
            1.0 — впереди столкновение;
            0.0 — безопасно.
        Используется как признак состояния
        '''
        head_x, head_y = self.snake[0] # координаты головы
        nx = head_x + self.direction[0] # Координаты следующей клетки
        ny = head_y + self.direction[1] # Координаты следующей клетки

        if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT: # Столкновение со стеной
            return 1.0
        if (nx, ny) in self.snake: # Столкновение с телом
            return 1.0
        return 0.0 # Безопасное движение

    def get_state(self):
        """Формирует вектор признаков состояния"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Базовые признаки
        state = [
            head_x / GRID_WIDTH,  # Нормализованная позиция X головы
            head_y / GRID_HEIGHT,  # Нормализованная позиция Y головы
            food_x / GRID_WIDTH,  # Нормализованная позиция X еды
            food_y / GRID_HEIGHT,  # Нормализованная позиция Y еды
            # Расстояние до еды
            abs(head_x - food_x) / GRID_WIDTH,
            abs(head_y - food_y) / GRID_HEIGHT,
            # Направление движения (one-hot encoding)
            1 if self.direction == UP else 0,
            1 if self.direction == DOWN else 0,
            1 if self.direction == LEFT else 0,
            1 if self.direction == RIGHT else 0,
            # Опасность впереди
            self._danger_ahead(),
            # Длина змейки (нормализованная)
            len(self.snake) / (GRID_WIDTH * GRID_HEIGHT),
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Шаг среды"""
        self.steps_without_food += 1

        # Изменение направления
        if action == 0:  # Вперед (не менять направление)
            pass
        elif action == 1:  # Направо
            if self.direction == UP:
                self.direction = RIGHT
            elif self.direction == RIGHT:
                self.direction = DOWN
            elif self.direction == DOWN:
                self.direction = LEFT
            elif self.direction == LEFT:
                self.direction = UP
        elif action == 2:  # Налево
            if self.direction == UP:
                self.direction = LEFT
            elif self.direction == LEFT:
                self.direction = DOWN
            elif self.direction == DOWN:
                self.direction = RIGHT
            elif self.direction == RIGHT:
                self.direction = UP

        # Движение
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Проверка столкновений
        if (
            new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
        ):
            self.done = True
            return self.get_state(), -10.0, self.done, {"score": self.score}

        if new_head in self.snake:
            self.done = True
            return self.get_state(), -10.0, self.done, {"score": self.score}

        # Добавление новой головы
        self.snake.insert(0, new_head)

        # Проверка съедания еды
        reward = -0.01  # Маленький штраф за каждый шаг
        if new_head == self.food:
            self.score += 1
            self.food = self._generate_food()
            reward = 10.0  # Большая награда за еду
            self.steps_without_food = 0
        else:
            # Удаляем хвост только если не съели еду
            self.snake.pop()

        # Штраф за слишком долгое блуждание без еды
        if self.steps_without_food > GRID_WIDTH * GRID_HEIGHT * 2:
            reward = -5.0
            self.done = True

        # Награда за приближение к еде
        old_head = (head_x, head_y)
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        if new_dist < old_dist:
            reward += 0.1  # Небольшая награда за приближение к еде
        elif new_dist > old_dist:
            reward -= 0.1  # Небольшой штраф за удаление от еды

        return self.get_state(), reward, self.done, {"score": self.score}

    def render(self):
        """Визуализация игры"""
        if self.render_mode != "human":
            return

        self.screen.fill((0, 0, 0))

        # Рисуем сетку
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(
                    x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
                )
                pygame.draw.rect(self.screen, (30, 30, 30), rect, 1)

        # Рисуем змейку
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0, 150, 0), rect, 2)

            # Глаза у головы
            if i == 0:
                eye_size = BLOCK_SIZE // 5
                # Левый глаз
                eye_x = x * BLOCK_SIZE + BLOCK_SIZE // 3
                eye_y = y * BLOCK_SIZE + BLOCK_SIZE // 3
                pygame.draw.circle(
                    self.screen, (255, 255, 255), (eye_x, eye_y), eye_size
                )
                # Правый глаз
                eye_x = x * BLOCK_SIZE + 2 * BLOCK_SIZE // 3
                pygame.draw.circle(
                    self.screen, (255, 255, 255), (eye_x, eye_y), eye_size
                )

        # Рисуем еду
        food_rect = pygame.Rect(
            self.food[0] * BLOCK_SIZE,
            self.food[1] * BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE,
        )
        pygame.draw.rect(self.screen, (255, 0, 0), food_rect)
        pygame.draw.circle(
            self.screen,
            (255, 100, 100),
            (
                self.food[0] * BLOCK_SIZE + BLOCK_SIZE // 2,
                self.food[1] * BLOCK_SIZE + BLOCK_SIZE // 2,
            ),
            BLOCK_SIZE // 3,
        )

        # Отображаем счет
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)
