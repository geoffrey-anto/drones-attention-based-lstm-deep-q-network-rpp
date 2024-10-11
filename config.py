MAP_SIZE = 20  # km

START_WAYPOINT = (0, 3, 0.6)  # (x, y, z) km
END_WAYPOINT = (20, 20, 0.6)  # (x, y, z) km

SPEED = 0.2  # km/s
YAW_RATE = 3  # deg/s
TIME_INTERVAL = 1  # s

MAXIMUM_LATERAL_OVERLOAD = 0.9  # g
MINIMUM_SAFETY_DISTANCE = 0.2  # km

STATE_SIZE = 15
SEQUENCE_LEN = 3
INPUT_SHAPE = (SEQUENCE_LEN, STATE_SIZE)

ACTION_SIZE = 3

REWARD_MAP = {
    "collision":-10,
    "arrival": 20,
    "track_angle": 5,
    "distance": 5,
    "gamma": 0.8,
    "alpha": 0.8,
    "Dsafe": 0.2
}

NUM_OBSTACLES = 12
