import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Attention, Input
from collections import deque
from tensorflow.keras.models import Model
from ExperienceReplayBuffer import ExperienceReplayBuffer
import numpy as np
from StateHistoryBuffer import StateHistoryBuffer
from config import SEQUENCE_LEN, STATE_SIZE, REWARD_MAP, ACTION_SIZE


class DQNALSTM:

    def __init__(self, input_shape):
        self.memory = ExperienceReplayBuffer(1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model(input_shape, ACTION_SIZE)
        self.target_model = self.build_model(input_shape, ACTION_SIZE)
        self.update_target_model()

    def build_model(self, input_shape, action_size):
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        attention_out = Attention()([lstm_out, lstm_out])
        lstm_out2 = LSTM(32)(attention_out)
        outputs = Dense(action_size, activation='linear')(lstm_out2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward):
        self.memory.add(state, action, reward)
        
    def act(self, state: StateHistoryBuffer):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(ACTION_SIZE)
        else:
            return np.argmax(self.model.predict(state.get_state())[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = self.memory.get_random_samples(batch_size)
        
        for item in minibatch:
            state = item.get_prev_state()
            state = state.reshape((1, SEQUENCE_LEN, STATE_SIZE))
            
            target = self.model.predict(state)
            
            if item.rt == REWARD_MAP["arrival"]:
                target[0][item.at] = item.rt
            else:
                next_state = item.get_state()
                next_state = next_state.reshape((1, SEQUENCE_LEN, STATE_SIZE))
                
                next_q = self.target_model.predict(next_state)[0]
                target[0][item.at] = item.rt + self.gamma * np.amax(next_q)
                
            self.model.fit(state, target, epochs=1)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.update_target_model()

    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)

# if __name__ == "__main__":
#     # Create an instance of DQNALSTM
#     dqn = DQNALSTM(input_shape=(SEQUENCE_LEN, STATE_SIZE))

#     # Test the act() method
#     state = StateHistoryBuffer()
#     action = dqn.act(state)
#     print("Random action:", action)

#     # Test the remember() method
#     state = [0.1] * 15
#     action = 0
#     reward = 1
#     for i in range(40):
#         dqn.remember(state, action, reward)

#     # Test the replay() method
#     batch_size = 32
#     dqn.replay(batch_size)

#     # Test the load() and save() methods
#     model_name = "dqn_model.h5"
#     dqn.save(model_name)
#     dqn.load(model_name)

#     # Test the update_target_model() method
#     dqn.update_target_model()
