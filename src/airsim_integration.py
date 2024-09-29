import airsim
import numpy as np
from src.lstm_model import DQNLSTM
from src.safety_checker import SafetyChecker

class DroneMOAAPP:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.client = airsim.MultirotorClient(ip="192.168.1.4")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.state_history = []
        self.safety_checker = SafetyChecker(self.client, config)

    def get_state(self):
        """Get the current state of the UAV (position, velocity)"""
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        velocity = state.kinematics_estimated.linear_velocity
        return np.array([
            position.x_val, position.y_val, position.z_val,
            velocity.x_val, velocity.y_val, velocity.z_val
        ])

    def predict_next_action(self, state):
        """Predict the next action using the RL model"""
        state = state.reshape((1, self.config.SEQ_LENGTH, 6))  # Reshape to fit model input
        action = self.model.act(state)
        return action

    def move_drone(self, action):
        """Move the drone based on the action"""
        waypoint = self.config.ACTION_MAP[action]  # Map the action to a movement command
        if self.safety_checker.is_safe_move(waypoint):
            self.client.moveToPositionAsync(waypoint[0], waypoint[1], waypoint[2], 5).join()
        else:
            print("Unsafe move detected. Executing collision avoidance.")
            self.safety_checker.avoid_collision()

    def run_mission(self, mission):
        """Run the mission (start with takeoff, navigate to waypoints, land)"""
        self.takeoff(mission['takeoff'])

        for waypoint in mission['waypoints']:
            print(f"Moving to waypoint: {waypoint}")
            for step in range(self.config.MAX_STEPS):
                current_state = self.get_state()
                self.state_history.append(current_state)
                if len(self.state_history) < self.config.SEQ_LENGTH:
                    continue

                # Get the next action from the model
                state = np.array(self.state_history[-self.config.SEQ_LENGTH:])
                action = self.predict_next_action(state)
                self.move_drone(action)

        self.land(mission['landing'])
        print("Mission complete.")

    def takeoff(self, location):
        """Takeoff at the specified location"""
        self.client.takeoffAsync().join()
        self.move_drone(location)

    def land(self, location):
        """Land at the specified location"""
        self.client.landAsync().join()

    def cleanup(self):
        """Clean up the UAV after the mission"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
