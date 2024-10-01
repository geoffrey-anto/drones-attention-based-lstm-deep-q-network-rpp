class Config:
    # Data generation
    NUM_SAMPLES = 10000
    SEQ_LENGTH = 50

    # Model training
    EPOCHS = 100
    BATCH_SIZE = 32

    # Action mapping
    ACTION_MAP = {
        0: [0, 0, -3],  # Move forward
        1: [3, 0, 0],  # Move right
        2: [-3, 0, 0],  # Move left
        3: [0, 0, 3],  # Move backward
    }

    # AirSim
    MAX_STEPS = 1000
    DISTANCE_THRESHOLD = 2.0
    STEP_DELAY = 0.1
    SAFE_DISTANCE = 5.0

    # Simulation
    NUM_OBSTACLES = 10
    
    IP = "192.168.0.100"
