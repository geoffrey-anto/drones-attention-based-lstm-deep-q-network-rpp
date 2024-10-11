import numpy as np


class StateHistoryBuffer():

    def __init__(self) -> None:
        self.st_m2 = [0.0] * 15
        self.st_m1 = [0.0] * 15
        self.st = [0.0] * 15
        self.st_p1 = [0.0] * 15
        
        self.at_m2 = 0.0
        self.at_m1 = 0.0
        self.at = 0.0
        
        self.rt = 0.0
        
    def get(self):
        return [self.st_m2, self.st_m1, self.st, self.st_p1, self.at_m2, self.at_m1, self.at, self.rt]
    
    def get_state(self):
        return np.array([self.st_m1, self.st, self.st_p1])

    def get_prev_state(self):
        return np.array([self.st_m2, self.st_m1, self.st])
    
    def get_next_state_history(self, next_state, action, reward):
        next_state_history = StateHistoryBuffer()
        
        next_state_history.st_m2 = self.st_m1
        next_state_history.st_m1 = self.st
        next_state_history.st = self.st_p1
        next_state_history.st_p1 = next_state
        
        next_state_history.at_m2 = self.at_m1
        next_state_history.at_m1 = self.at
        next_state_history.at = action
        
        next_state_history.rt = reward
        
        return next_state_history
    
    def __str__(self) -> str:
        return f"[st_m2: {self.st_m2}, st_m1: {self.st_m1}, st: {self.st}, st_p1: {self.st_p1}, at_m2: {self.at_m2}, at_m1: {self.at_m1}, at: {self.at}, rt: {self.rt}]"
    
    def __repr__(self) -> str:
        return self.__str__()
