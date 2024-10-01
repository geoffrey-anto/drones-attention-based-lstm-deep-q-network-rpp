import torch.nn as nn
import torch
from collections import deque
import random
import numpy as np


class AttentionWithLSTM(nn.Module):

    def __init__(self, input_shape, action_size, *args, **kwargs) -> None:
        super(AttentionWithLSTM, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.action_size = action_size
        
        self.lstm1 = nn.LSTM(input_shape[-1], 64, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        
        self.linear = nn.Linear(32, action_size)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.attention(x, x, x)
        x, _ = self.lstm2(x)
        x = self.linear(x[:, -1,:])
        return x


class DQNLSTM:

    def __init__(self, input_shape, action_size, *args, **kwargs):
        super(DQNLSTM, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.action_size = action_size
        
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = AttentionWithLSTM(input_shape, action_size)
        self.target_model = AttentionWithLSTM(input_shape, action_size)
        
        self.update_target_model()
        
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def next(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            
            self.optim.zero_grad()
            
            target = self.model(state)
            
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * torch.max(self.target_model(next_state)[0])
            
            loss = self.loss_fn(self.model(state), target)
            
            loss.backward()
            
            self.optim.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.update_target_model()

    def load_model(self, name):
        self.model.load_state_dict(torch.load(name))
    
    def save_model(self, name):
        torch.save(self.model.state_dict(), name)


if __name__ == '__main__':
    model = DQNLSTM((1, 4, 4), 2)
    model.save_model('model.pth')
    model.load_model('model.pth')
    model.next(torch.rand(1, 4, 4))
    
    res = model.next(torch.rand(1, 4, 4))
    
    model.remember(torch.rand(1, 4, 4), res, 1, torch.rand(1, 4, 4), False)
    
    print(res)
    
    model.replay(1)
