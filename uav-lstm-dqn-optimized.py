import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import math
from collections import deque
import argparse


# LSTM-DQN Network
class LSTMDeepQNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMDeepQNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x[:, -1,:])
        return x, hidden


# UAV Class
class UAV:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 2
        self.max_turn_angle = math.radians(10)

    def move(self, action):
        # Adjust the UAV's angle based on action
        self.angle += (action - 1) * self.max_turn_angle
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

        # Ensure UAV stays within field boundaries
        self.x = max(0, min(self.x, 800))
        self.y = max(0, min(self.y, 800))


# Environment setup
def setup_environment():
    uav = UAV(50, 50)
    obstacles = [{'x': random.randint(100, 700), 'y': random.randint(100, 700)} for _ in range(5)]
    target = {'x': 750, 'y': 750}
    return uav, obstacles, target


# Helper functions
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_state(uav, obstacles, target):
    state = [
        uav.x / 800, uav.y / 800,
        distance(uav.x, uav.y, target['x'], target['y']) / math.sqrt(800 ** 2 + 800 ** 2),
        math.sin(uav.angle), math.cos(uav.angle)
    ]
    for obstacle in obstacles:
        state.append(distance(uav.x, uav.y, obstacle['x'], obstacle['y']) / math.sqrt(800 ** 2 + 800 ** 2))
    return np.array(state)


# DQN Agent
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = LSTMDeepQNetwork(state_size, 24, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Reshape for LSTM
        act_values, _ = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
            target = reward
            if not done:
                next_q_values, _ = self.model(next_state)
                target = reward + self.gamma * torch.max(next_q_values).item()
            q_values, _ = self.model(state)
            target_f = q_values.clone()
            target_f[0, action] = target
            loss = self.criterion(q_values, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()


# Training function
def train_agent(episodes, batch_size):
    state_size = 10  # Update based on the state representation
    action_size = 3  # Left, Right, Forward
    agent = DQNAgent(state_size, action_size)
    training_log = []

    for e in range(episodes):
        uav, obstacles, target = setup_environment()
        state = get_state(uav, obstacles, target)
        for time in range(500):
            action = agent.act(state)
            uav.move(action)
            next_state = get_state(uav, obstacles, target)
            reward = -0.01  # Default small penalty
            done = False

            # Collision detection and reward assignment
            if any(distance(uav.x, uav.y, obs['x'], obs['y']) < 25 for obs in obstacles):
                reward = -10
                done = True
                # Move UAV in the opposite direction upon collision
                uav.angle += math.pi  # Reverse the UAV's direction
                uav.move(1)  # Move one step in the opposite direction
            elif distance(uav.x, uav.y, target['x'], target['y']) < 15:
                reward = 20
                done = True

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                episode_info = f"episode: {e}/{episodes}, score: {time}, epsilon: {agent.epsilon:.2}"
                print(episode_info)
                training_log.append(episode_info)
                with open("training_log.txt", "a") as log_file:
                    log_file.write(episode_info + "\n")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    return agent


# Pygame visualization
def visualize_agent(agent):
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("UAV Path Planning Simulation")

    uav, obstacles, target = setup_environment()
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = get_state(uav, obstacles, target)
        action = agent.act(state)
        uav.move(action)

        if any(distance(uav.x, uav.y, obs['x'], obs['y']) < 25 for obs in obstacles):
            print("Collision! Mission failed.")
            running = False
        elif distance(uav.x, uav.y, target['x'], target['y']) < 15:
            print("Target reached! Mission successful.")
            running = False

        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (0, 0, 255), (int(uav.x), int(uav.y)), 5)
        for obstacle in obstacles:
            pygame.draw.circle(screen, (255, 0, 0), (obstacle['x'], obstacle['y']), 20)
        pygame.draw.circle(screen, (0, 255, 0), (target['x'], target['y']), 10)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Path Planning with LSTM-DQN")
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--visualize', action='store_true', help='Visualize the trained agent')
    args = parser.parse_args()

    if args.train:
        agent = train_agent(episodes=50, batch_size=32)
        agent.save('uav_lstm_dqn_model.pth')
        print("Training completed and model saved.")
    
    if args.visualize:
        agent = DQNAgent(10, 3)
        agent.load('uav_lstm_dqn_model.pth')
        visualize_agent(agent)
