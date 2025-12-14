from train import train_agent
from utils import plot_results, test_agent


def main():
    EPISODES = 500
    RENDER_EVERY = 25

    try:
        agent, scores, losses, epsilons = train_agent(
            episodes=EPISODES, render_every=RENDER_EVERY
        )
        plot_results(scores, losses, epsilons)
        test_scores = test_agent(agent, episodes=5)
    except KeyboardInterrupt:
        print("Обучение прервано пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        import pygame

        pygame.quit()


if __name__ == "__main__":
    main()
