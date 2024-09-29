
# MOAAPP - Multi-Objective Attention-Based Adaptive Path Planner (UAV Path Planning) 🚀

This project implements a **UAV path planning system** using **Deep Q-Learning (DQN)** combined with **LSTM** and **Attention mechanisms** for reinforcement learning. It leverages the AirSim simulation environment to enhance the UAV's ability to navigate in complex, obstacle-rich environments, optimizing multiple objectives like path length, safety, and energy efficiency.

## Key Features

- **Deep Q-Learning with LSTM and Attention**: A reinforcement learning model that adapts UAV path planning by learning the best actions through an attention-enhanced LSTM network.
- **Multi-Objective Optimization**: Simultaneously optimizes for multiple goals like minimizing path length, avoiding obstacles, and maximizing energy efficiency.
- **Dynamic Attention Mechanism**: Attention layers help the model focus on the most relevant environmental features during the decision-making process.
- **Integration with AirSim**: Uses Microsoft's AirSim platform to simulate complex UAV navigation environments.
- **Safety Checker & Collision Avoidance**: Monitors and prevents collisions using lidar data and obstacle detection.

## Project Structure

```plaintext
project_root/
│
├── src/
│   ├── lstm_model.py           # LSTM with Attention and DQN for path planning
│   ├── airsim_integration.py   # Integration with AirSim for UAV control
│   ├── data_generation.py      # Generates training data (optional)
│   ├── mission_planner.py      # Collects user input for mission planning
│   ├── simulation.py           # Simulation setup and execution
│   └── safety_checker.py       # Monitors collisions and handles obstacle avoidance
│
├── config/
│   └── config.py               # Configuration settings for the project
│
├── main.py                     # Main script to run the UAV system
├── requirements.txt            # Python dependencies
└── README.md                   # Project information and setup instructions
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saswatakumardash/MOAAPP-PathPlanner.git
   cd MOAAPP-PathPlanner
   ```

2. **Install dependencies**:
   Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up AirSim**:
   - Follow the official [AirSim documentation](https://microsoft.github.io/AirSim/) to install and set up an AirSim environment.

## Running the Project

### Step 1: Prepare your AirSim environment
Ensure your AirSim environment is properly configured and running.

### Step 2: Run the main script
Execute the following command to start the system:
```bash
python main.py
```

### Step 3: Mission planning
You will be prompted to input the **takeoff location**, **waypoints**, and **landing location** during the mission planning phase.

### Step 4: Observe the simulation
Watch the UAV navigate through the environment while avoiding obstacles and completing the mission.

## Configuration

You can modify the project settings in `config/config.py` to adjust simulation parameters:

- **NUM_SAMPLES**: Number of samples for data generation.
- **SEQ_LENGTH**: Sequence length for LSTM training.
- **EPOCHS**: Number of epochs for training.
- **BATCH_SIZE**: Batch size used in training.
- **MAX_STEPS**: Maximum steps allowed for the UAV to reach a waypoint.
- **SAFE_DISTANCE**: Minimum safe distance from obstacles for collision avoidance.
- **ACTION_MAP**: Defines the UAV's movement actions, such as moving forward, backward, left, right.

## Deep Q-Learning (DQN) Integration

The core of this project uses **Deep Q-Learning (DQN)**, which enables the UAV to learn an optimal path by interacting with the environment:

- **Experience Replay Buffer**: Stores previous experiences (state, action, reward, next state) to break correlation between consecutive learning samples.
- **Target and Policy Networks**: Two separate neural networks are used to stabilize training — the **target network** is periodically updated with weights from the **policy network**.
- **Epsilon-Greedy Action Selection**: Balances exploration and exploitation during training.

### Model Architecture:

1. **LSTM + Attention**: 
   - The model first processes sequential state data with LSTM layers and a dynamic attention mechanism to focus on relevant environmental features.
   - The output of the LSTM-attention network is fed into a **Dense layer** representing Q-values for possible actions.

2. **DQN Logic**: 
   - The agent learns the optimal path by predicting Q-values and updating them through the Q-learning update rule: 
     \[
     Q(s, a) \leftarrow r + \gamma \max(Q(s', a'))
     \]
   - Where `r` is the reward, `\gamma` is the discount factor, `s` is the state, and `a` is the action.

## Future Enhancements

- **Multi-UAV Coordination**: Add support for multi-agent collaborative path planning.
- **Real-World Application**: Deploy the model for real-world UAV navigation tasks.
- **Advanced Attention Mechanisms**: Explore transformer-based attention for further improvements in complex environments.

## Authors

- **Saswata Kumar Dash** - [LinkedIn](https://www.linkedin.com/in/saswata-kumar-dash)
- **Geoffrey Anto Ignatius E** - [LinkedIn](https://www.linkedin.com/in/geoffrey-anto)

---

This project is a part of ongoing research in adaptive UAV navigation using reinforcement learning, aiming to improve the adaptability and efficiency of UAVs in complex environments.
