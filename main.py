from src.Environment import Environment
from src.Simulation import Simulation
import argparse
from src.Config import MAP_SIZE, SEQUENCE_LEN, STATE_SIZE
from src.DQNLSTM import DQNLSTM
from src.DQNALSTM import DQNALSTM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the model", action="store_true")
    parser.add_argument("--test", help="Test the model", action="store_true")
    parser.add_argument("--model", help="Model Weights Location", action="store")
    parser.add_argument("--model_type", help="Model Type", action="store")
    args = parser.parse_args()
    
    if args.model_type not in ["attention", "lstm"]:
        print("Please provide a valid model type")
        exit()

    if args.train:
        env = Environment(MAP_SIZE, MAP_SIZE, 0.6, train=True)
        model = None
        
        if args.model_type == "attention":
            model = DQNALSTM(input_shape=(SEQUENCE_LEN, STATE_SIZE))
        else:
            model = DQNLSTM(input_shape=(SEQUENCE_LEN, STATE_SIZE))
        
        sim = Simulation(env, model)
        sim.run_train()
    elif args.test:
        env = Environment(MAP_SIZE, MAP_SIZE, 0.6, train=False)
        if args.model_type == "attention":
            model = DQNALSTM(input_shape=(SEQUENCE_LEN, STATE_SIZE))
        else:
            model = DQNLSTM(input_shape=(SEQUENCE_LEN, STATE_SIZE))
        model.model.summary()
        sim = Simulation(env, model)
        sim.run_test(args.model)
    else:
        print("Please provide either --train or --test argument")
