import numpy as np
import pygame
from env import SnakeGame
from agent import DQNAgent


def train_agent(episodes=1000, render_every=50):
    """Функция обучения с визуализацией"""
    env = SnakeGame(render_mode="human")
    state_dim = len(env.get_state())
    action_dim = 3  # Вперед, направо, налево
    agent = DQNAgent(state_dim, action_dim)

    scores = []  # итоговый счёт за эпизод
    losses = []  # средняя функция потерь
    epsilons = []  # значение ε

    print("Начало обучения...")
    print(f"Размер состояния: {state_dim}")
    print(f"Количество действий: {action_dim}")

    for episode in range(
        episodes
    ):  # Эпизод — один полный запуск игры от reset до done=True
        state = env.reset()  # Сброс среды
        total_reward = 0  # суммарная награда
        episode_loss = 0  # накопленная ошибка
        steps = 0  # число шагов в эпизоде

        # Обработка событий Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        while True:
            # Выбор действия
            action = agent.choose_action(state)

            # Выполнение действия: next_state — новое состояние, reward — полученная награда, done — флаг завершения эпизода, info - счет
            next_state, reward, done, info = env.step(action)

            # Сохранение опыта
            agent.store_experience(state, action, reward, next_state, done)

            # Обучение (не на каждом шаге)
            if steps % 4 == 0:
                loss = agent.train()
                if loss:
                    episode_loss += loss

            # Обновление состояния
            state = next_state
            total_reward += reward
            steps += 1

            # Визуализация
            if episode % render_every == 0:  # Рендеринг только некоторых эпизодов:
                env.render()
                pygame.display.set_caption(
                    f"Змейка DQN - Эпизод: {episode}, "
                    f"Счет: {info['score']}, "
                    f"Epsilon: {agent.epsilon:.3f}"
                )

            if done:
                break

        # Сбор статистики
        scores.append(info["score"])  # Сохранение итогового счёта
        losses.append(episode_loss / max(steps, 1))  # Средняя ошибка за эпизод
        epsilons.append(agent.epsilon)  # Запоминаем значение ε

        # Вывод прогресса
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])  # Средний счёт за последние 10 эпизодов
            print(
                f"Эпизод: {episode+1:4d}/{episodes} | "
                f"Счет: {info['score']:2d} | "
                f"Средний счет (10 эп.): {avg_score:5.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Память: {len(agent.memory)}"
            )

    return agent, scores, losses, epsilons
