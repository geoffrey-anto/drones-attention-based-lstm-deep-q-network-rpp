import random
from src.StateHistoryBuffer import StateHistoryBuffer
from typing import List


class ExperienceReplayBuffer():

    def __init__(self, queue_max_size=1000) -> None:
        self._queue: List[StateHistoryBuffer] = [] 
        self._queue_max_size = queue_max_size
        
    def add(self, state, action, reward):
        if len(self._queue) == 0:
            self._queue.append(StateHistoryBuffer().get_next_state_history(state, action, reward))
            return
        elif len(self._queue) >= self._queue_max_size:
            self._queue.pop(0)
        
        self._queue.append(self._queue[-1].get_next_state_history(state, action, reward))
    
    def clear(self):
        self._queue.clear()
        
    def get_random_samples(self, batch_size=10):
        return random.choices(self._queue, k=batch_size)
    
    def print(self):
        print(self._queue, len(self._queue))
        
    def __len__(self):
        return len(self._queue)
