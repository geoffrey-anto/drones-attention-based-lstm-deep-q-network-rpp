import airsim
import numpy as np


class SafetyChecker:

    def __init__(self, client, config):
        self.client = client
        self.config = config

    def is_safe_move(self, waypoint):
        # Check for collisions
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return False

        # Check for proximity to obstacles
        lidar_data = self.client.getLidarData(lidar_name="LidarSensor1", vehicle_name="Drone1")
        if lidar_data.point_cloud:
            points = np.array(lidar_data.point_cloud).reshape((-1, 3))
            distances = np.linalg.norm(points, axis=1)
            if np.any(distances < self.config.SAFE_DISTANCE):
                return False

        return True

    def avoid_collision(self):
        # Simple collision avoidance: move upwards
        current_position = self.client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position
        safe_position = airsim.Vector3r(current_position.x_val, current_position.y_val, current_position.z_val - 5)
        self.client.moveToPositionAsync(safe_position.x_val, safe_position.y_val, safe_position.z_val, 5).join()
