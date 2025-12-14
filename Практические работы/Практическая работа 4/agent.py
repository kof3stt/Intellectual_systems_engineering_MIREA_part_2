import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import QNetwork


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        '''
        Docstring для __init__
        
        :param state_dim: размер вектора состояния
        :param action_dim: количество возможных действий
        '''
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = QNetwork(state_dim, action_dim) # Основная сеть, используется для выбора действий, веса обновляются при обучении
        self.target_net = QNetwork(state_dim, action_dim) # Целевая сеть, используется для расчёта целевого значения Q_target, обеспечивает стабильность обучения
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Копирование весов

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001) # Оптимизатор Adam
        self.loss_fn = nn.MSELoss() # Функция потерь

        # Гиперпараметры обучения
        self.gamma = 0.99  # Коэффициент дисконтирования (учитывает будущие награды)
        self.epsilon = 1.0  # Начальный epsilon (100% случайных действий, стимулирует исследование среды)
        self.epsilon_min = 0.01 # гарантирует небольшую долю случайности
        self.epsilon_decay = 0.995 # коэффициент для экспоненциального уменьшения epsilon
        self.batch_size = 64 # Размер мини-батча
        self.target_update_freq = 10  # Частота обновления целевой сети

        # Ограниченная память
        self.memory = deque(maxlen=20000) # Буфер опыта
        self.train_step_counter = 0 # Счётчик шагов обучения

    def choose_action(self, state, training=True):
        """Выбор действия с epsilon-greedy стратегией"""
        if training and np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1) # Случайное действие из допустимых

        state_tensor = torch.FloatTensor(state).unsqueeze(0) # Преобразование состояния
        with torch.no_grad(): # Отключение вычисления градиентов
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item() # Выбираем действие с максимальной ценностью

    def store_experience(self, state, action, reward, next_state, done):
        """Сохранение опыта в память"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Обучение на мини-батче из памяти"""
        if len(self.memory) < self.batch_size: # Обучение начинается только при достаточном опыте
            return 0

        # Выбор случайного мини-батча
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Конвертация в тензоры
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Текущие Q-значения
        current_q = self.policy_net(states).gather(1, actions).squeeze()

        # Следующие Q-значения (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True) # Выбор действий по policy-сети
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze() # Оценка действий по target-сети
            target_q = rewards + (1 - dones) * self.gamma * next_q # Формула Беллмана

        # Вычисление потерь
        loss = self.loss_fn(current_q, target_q)

        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Ограничение градиентов
        self.optimizer.step() # Обновление весов сети

        # Уменьшение epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Обновление целевой сети
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Копирование весов

        return loss.item()
