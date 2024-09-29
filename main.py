from src.lstm_model import DQNLSTM
from src.airsim_integration import DroneMOAAPP
from src.simulation import Simulation
from config.config import Config

def main():
    # Load configuration settings
    config = Config()

    # Initialize Deep Q-Learning with LSTM and Attention model
    print("Initializing DQN-LSTM model...")
    action_size = len(config.ACTION_MAP)  # Number of possible actions
    dqn_lstm_model = DQNLSTM((config.SEQ_LENGTH, 6), action_size)

    # Initialize simulation environment
    print("Setting up simulation environment...")
    simulation = Simulation(config)
    simulation.setup_environment()

    # Initialize the drone control system (using RL)
    print("Initializing drone system...")
    drone = DroneMOAAPP(dqn_lstm_model, config)

    # Run the mission within the simulation
    print("Running the mission...")
    simulation.run_simulation(drone)

    # After the mission, save the trained model
    print("Saving trained model...")
    dqn_lstm_model.save("dqn_lstm_model.h5")

    # Clean up and reset the UAV system
    drone.cleanup()
    print("Mission complete, UAV system cleaned up.")

if __name__ == "__main__":
    main()
