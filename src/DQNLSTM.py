import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Attention, Input
from tensorflow.keras.models import Model
from src.ExperienceReplayBuffer import ExperienceReplayBuffer
import numpy as np
from src.StateHistoryBuffer import StateHistoryBuffer
from src.Config import SEQUENCE_LEN, STATE_SIZE, REWARD_MAP, ACTION_SIZE
from tensorflow.keras.layers import MultiHeadAttention
import pickle


class DQNLSTM:

    def __init__(self, input_shape):
        self.memory = ExperienceReplayBuffer(100)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model(input_shape, ACTION_SIZE)
        self.target_model = self.build_model(input_shape, ACTION_SIZE)
        self.update_target_model()

    def build_model(self, input_shape, action_size):
        inputs = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(inputs)
        # x = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
        x = LSTM(64)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(action_size, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward):
        self.memory.add(state, action, reward)
        
    def act(self, state: StateHistoryBuffer, test=False):
        if test:
            if np.random.rand() <= 0.15:
                return np.random.choice(ACTION_SIZE)
            curr = state.get_state()
            curr = curr.reshape((1, SEQUENCE_LEN, STATE_SIZE))
            pred = self.model.predict(curr, verbose=0)[0]
            return np.argmax(pred)
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(ACTION_SIZE)
        else:
            curr = state.get_state()
            curr = curr.reshape((1, SEQUENCE_LEN, STATE_SIZE))
            pred = self.model.predict(curr, verbose=0)[0]
            return np.argmax(pred)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = self.memory.get_random_samples(batch_size)
        
        for item in minibatch:
            state = item.get_prev_state()
            state = np.reshape(state, (1, SEQUENCE_LEN, STATE_SIZE))
            
            target = self.model.predict(state, verbose=0)
            print(f"Target: {target}", end=" ")
            
            if item.rt == REWARD_MAP["arrival"]:
                target[0][item.at] = item.rt
            else:
                next_state = item.get_state()
                next_state = np.reshape(next_state, (1, SEQUENCE_LEN, STATE_SIZE))
                next_q = self.target_model.predict(next_state, verbose=0)[0]
                target[0][item.at] = item.rt + self.gamma * np.max(next_q)
            
            print(f"Updated: {target}")
            
            self.model.fit(state, target, epochs=1, verbose=0) 

        print(f"Replaying {batch_size} samples with epsilon {self.epsilon}")
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.update_target_model()

    def load(self, name):
        self.load_model_pickle(name)
        
    def save(self, name):
        self.save_model_pickle(self.model, name)
            
    def save_model_pickle(self, model, name):
        with open(name, 'wb') as f:
            pickle.dump(model, f)

    def load_model_pickle(self, name):
        with open(name, 'rb') as f:
            self.model = pickle.load(f)


if __name__ == "__main__":
    model = DQNLSTM((SEQUENCE_LEN, STATE_SIZE))
    model.model.summary()
