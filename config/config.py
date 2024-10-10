class Config:
    # Data generation
    NUM_SAMPLES = 10000
    SEQ_LENGTH = 50

    # Model training
    EPOCHS = 100
    BATCH_SIZE = 32
    
    ACTION_STEP_SIZE_XY = 2
    ACTION_STEP_SIZE_Z = 0

    # Action mapping
    ACTION_MAP = {
        0: [+ACTION_STEP_SIZE_XY, 0, 0],  # Move Forward (+X)
        1: [-ACTION_STEP_SIZE_XY, 0, 0],  # Move Backward (-X)
        2: [0, -ACTION_STEP_SIZE_XY, 0],  # Move Left (-Y)
        3: [0, +ACTION_STEP_SIZE_XY, 0],  # Move Right (+Y)
        4: [0, 0, +ACTION_STEP_SIZE_Z],  # Move Up (+Z)
        5: [0, 0, -ACTION_STEP_SIZE_Z]  # Move Down (-Z)
    }

    # AirSim
    MAX_STEPS = 1000
    DISTANCE_THRESHOLD = 2.0
    STEP_DELAY = 0.1
    SAFE_DISTANCE = 5.0

    # Simulation
    NUM_OBSTACLES = 10
    
    IP = "192.168.1.4"
