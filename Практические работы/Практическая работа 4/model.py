import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Конструктор

        :param input_dim: размер вектора состояния
        :param output_dim: количество действий (вперед, вправо, влево)
        """
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, state):  # прямой проход сети
        return self.fc(state)  # Прогон состояния через все слои
