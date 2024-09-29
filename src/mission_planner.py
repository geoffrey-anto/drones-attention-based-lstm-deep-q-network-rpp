import numpy as np

class MissionPlanner:
    def __init__(self, config):
        self.config = config

    def plan_mission(self):
        print("\nMission Planner")
        print("---------------")
        
        # Select takeoff location
        takeoff_location = self.select_location("Enter takeoff location (x y z): ")
        
        # Select landing location
        landing_location = self.select_location("Enter landing location (x y z): ")
        
        # Select waypoints
        waypoints = []
        while True:
            waypoint = self.select_location("Enter waypoint (x y z) or 'done' to finish: ")
            if waypoint is None:
                break
            waypoints.append(waypoint)
        
        return {
            'takeoff': takeoff_location,
            'landing': landing_location,
            'waypoints': waypoints
        }

    def select_location(self, prompt):
        while True:
            try:
                user_input = input(prompt)
                if user_input.lower() == 'done':
                    return None
                coords = list(map(float, user_input.split()))
                if len(coords) != 3:
                    raise ValueError
                return np.array(coords)
            except ValueError:
                print("Invalid input. Please enter three numbers separated by spaces.")
