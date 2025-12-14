import matplotlib.pyplot as plt
import numpy as np
import pygame
from env import SnakeGame


def plot_results(scores, losses, epsilons):
    """Визуализация результатов обучения"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # График счетов
    axes[0].plot(scores, alpha=0.6)
    window = 50
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
        axes[0].plot(
            range(window - 1, len(scores)),
            moving_avg,
            "r-",
            linewidth=2,
            label=f"Среднее ({window} эп.)",
        )
    axes[0].set_title("Счет по эпизодам")
    axes[0].set_xlabel("Эпизод")
    axes[0].set_ylabel("Счет")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График потерь
    axes[1].plot(losses, alpha=0.6)
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window) / window, mode="valid")
        axes[1].plot(
            range(window - 1, len(losses)),
            moving_avg,
            "r-",
            linewidth=2,
            label=f"Среднее ({window} эп.)",
        )
    axes[1].set_title("Потери при обучении")
    axes[1].set_xlabel("Эпизод")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # График epsilon
    axes[2].plot(epsilons, "g-")
    axes[2].set_title("Epsilon по эпизодам")
    axes[2].set_xlabel("Эпизод")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def test_agent(agent, episodes=10):
    """Тестирование обученного агента"""
    env = SnakeGame(render_mode="human")
    test_scores = []

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, done, info = env.step(action)
            env.render()
            pygame.display.set_caption(
                f"Тестирование - Эпизод: {episode+1}, " f"Счет: {info['score']}"
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        test_scores.append(info["score"])
        print(f"Эпизод {episode+1}: Счет = {info['score']}")

    return test_scores
