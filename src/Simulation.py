# from src.DQNALSTM import DQNALSTM
from src.Config import SEQUENCE_LEN, STATE_SIZE, BATCH_SIZE, MAX_STEPS, MAP_SIZE
from src.StateHistoryBuffer import StateHistoryBuffer
from tqdm import tqdm
from src.Environment import Environment
from src.DQNLSTM import DQNLSTM
from src.DQNALSTM import DQNALSTM
import time


class Simulation():

    def __init__(self, env: Environment, model: DQNLSTM | DQNALSTM):
        self.env = env
        self.model = model
        self.state = StateHistoryBuffer()

    def run_train(self):
        for i in tqdm(range(MAX_STEPS)):
            if i % 50 == 0:
                if i % 25 == 0:
                    self.model.save(f"checkpoints/dqn_model_{i}.pkl")
                self.env.show_environment()
                
            action = self.model.act(self.state, test=False)
            
            next_state, reward, done = self.env.step(action)
            print(f"Action: {action}, Reward: {reward}, State: {next_state}")
            
            self.model.remember(self.env.agent.get_state(), action, reward)
            
            self.state = self.state.get_next_state_history(next_state, action, reward)

            if done:
                self.state = StateHistoryBuffer()
                self.env.show_environment()
                self.env = Environment(MAP_SIZE, MAP_SIZE, 0.6, train=True)
                self.model.save("dqn_model.pkl")
                return

            self.model.replay(BATCH_SIZE)
        
        self.env.show_environment()
        self.model.save("dqn_model.pkl")

    def run_test(self, model_location):
        self.model.load(model_location)
        
        timeTaken = 0.0
        
        for _ in tqdm(range(300)):
            startTime = time.time()
            action = self.model.act(self.state, test=True)
            endTime = time.time()
            
            timeTaken += (endTime - startTime)
            
            next_state, reward, done = self.env.step(action)
            
            self.state = self.state.get_next_state_history(next_state, action, reward)
            
            if done:
                print(*["-" * 10], "Done", *["-" * 10])
                print(f"Steps Taken: {_ + 1}")
                print("Average Time Taken: ", timeTaken / (_ + 1))
                print(f"Minimum Distance: {self.env.agent.global_dm}")
                print(f"Obstacle Hit Count: {len(self.env.agent.collision_count)}")
                print("Distance From Obstacles During Journey: ", end="")
                print(*self.env.agent.min_distances_at_each_step, sep=",")
                print(*["-" * 24])
                break
            
        print(*["-" * 10], "Done", *["-" * 10])
        print(f"Steps Taken: {_ + 1}")
        print("Average Time Taken: ", timeTaken / (_ + 1))
        print(f"Minimum Distance: {self.env.agent.global_dm}")
        print(f"Obstacle Hit Count: {len(self.env.agent.collision_count)}")
        print("Distance From Obstacles During Journey: ", end="")
        print(*self.env.agent.min_distances_at_each_step, sep=",")
        print(*["-" * 24])
        self.env.show_environment()
            
