import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Attention, Input
from collections import deque
from tensorflow.keras.models import Model
import random


class DQNLSTM:

    def __init__(self, input_shape, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model(input_shape, action_size)
        self.target_model = self.build_model(input_shape, action_size)
        self.update_target_model()

    def build_model(self, input_shape, action_size):
        # Define input layers
        inputs = Input(shape=input_shape)

        # LSTM layer
        lstm_out = LSTM(64, return_sequences=True)(inputs)

        # Attention layer
        attention_out = Attention()([lstm_out, lstm_out])  # Attention requires a list of inputs

        # Another LSTM layer
        lstm_out2 = LSTM(32)(attention_out)

        # Output layer for Q-values
        outputs = Dense(action_size, activation='linear')(lstm_out2)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        """Copies the weights from the main model to the target model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explore: random action
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit: choose best action

    def replay(self, batch_size):
        """Train the model using experiences in memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Reshape state to match the model's expected input shape
            state = state.reshape((1, self.config.SEQ_LENGTH, 6))
            
            # Check the shape of next_state and pad if necessary
            if next_state.shape != (1, self.config.SEQ_LENGTH, 6):
                next_state_padded = np.zeros((1, self.config.SEQ_LENGTH, 6))
                next_state_padded[0, -next_state.shape[1]:,:] = next_state
                next_state = next_state_padded
            
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                next_q = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(next_q)
            
            self.model.fit(state, target, epochs=1, verbose=0)

    def load(self, name):
        """Loads a model from file"""
        self.model.load_weights(name)

    def save(self, name):
        """Saves the model to a file"""
        self.model.save_weights(name)
