from src.DQNALSTM import DQNALSTM
from src.Config import SEQUENCE_LEN, STATE_SIZE, BATCH_SIZE, MAX_STEPS
from src.StateHistoryBuffer import StateHistoryBuffer
from tqdm import tqdm
from src.Environment import Environment


class Simulation():

    def __init__(self, env: Environment):
        self.env = env
        self.model = DQNALSTM(input_shape=(SEQUENCE_LEN, STATE_SIZE))
        self.state = StateHistoryBuffer()

    def run_train(self):
        for i in tqdm(range(MAX_STEPS)):
            if i % 50 == 0:
                self.model.save(f"checkpoints/dqn_model_{i}.pkl")
                self.env.show_environment()
                
            action = self.model.act(self.state, test=False)
            
            next_state, reward, done = self.env.step(action)
            
            self.model.remember(self.env.agent.get_state(), action, reward)
            
            self.state = self.state.get_next_state_history(next_state, action, reward)

            if done:
                self.state = StateHistoryBuffer()
                self.env.show_environment()
                self.model.save("dqn_model.pkl")
                return

            self.model.replay(BATCH_SIZE)
        
        self.env.show_environment()
        self.model.save("dqn_model.pkl")

    def run_test(self):
        self.model.load("checkpoints/dqn_model_450.pkl")
        
        for _ in tqdm(range(MAX_STEPS * 2)):
            action = self.model.act(self.state, test=True)
            
            next_state, reward, done = self.env.step(action)
            print(f"Action: {action}, Reward: {reward}, State: {next_state}")
            
            self.state = self.state.get_next_state_history(next_state, action, reward)
            
            if done:
                break
        self.env.show_environment()
            
