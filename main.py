from src.Environment import Environment
from src.Simulation import Simulation
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the model", action="store_true")
    parser.add_argument("--test", help="Test the model", action="store_true")
    
    args = parser.parse_args()
    
    if args.train:
        env = Environment(20, 20, 0.6, train=True)
        sim = Simulation(env)
        sim.run_train()
    elif args.test:
        env = Environment(20, 20, 0.6, train=False)
        sim = Simulation(env)
        sim.run_test()
    else:
        print("Please provide either --train or --test argument")
