import airsim
import numpy as np
from src.mission_planner import MissionPlanner

class Simulation:
    def __init__(self, config):
        self.config = config
        self.client = airsim.MultirotorClient(ip="192.168.1.4")
        self.client.confirmConnection()

    def setup_environment(self):
        """Setup simulation environment: weather, obstacles, etc."""
        print("Setting up simulation environment...")
        self.add_obstacles()
        self.set_weather()

    def add_obstacles(self):
        """Add random obstacles in the environment."""
        for _ in range(self.config.NUM_OBSTACLES):
            position = np.random.uniform(-50, 50, 3).tolist()  # Random position
            rotation = airsim.Quaternionr(0, 0, 0, 1)  # Static rotation using Quaternion
            
            # Create a pose using Vector3 and Quaternion
            pose = airsim.Pose(airsim.Vector3r(*position), rotation)

            # Attempt to spawn the obstacle
            try:
                self.client.simSpawnObject(f"obstacle_{_}", "Cube", pose, scale=airsim.Vector3r(1, 1, 1))
            except Exception as e:
                print(f"Failed to spawn obstacle {_}: {e}")

    def set_weather(self):
        """Set random weather conditions in the simulation."""
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, np.random.uniform(0, 1))
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, np.random.uniform(0, 1))

    def run_simulation(self, drone):
        """Runs the UAV mission"""
        print("Starting simulation...")
        mission_planner = MissionPlanner(self.config)
        mission = mission_planner.plan_mission()

        # try:
        drone.run_mission(mission)
        # except Exception as e:
        #     print(f"Simulation error: {e}")
        # finally:
        #     drone.cleanup()

