import matplotlib.pyplot as plt
from config import YAW_RATE, TIME_INTERVAL, SPEED, NUM_OBSTACLES, START_WAYPOINT
import numpy as np
import math
import random as rnd


class DroneMOAAPP:
    _instance = None
    _initialized = False
    
    def __new__(cls, *_, **__):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, start_position, target_position, obstacles) -> None:
        if not self._initialized:
            self.position = start_position
            self.target_position = target_position
            
            self.obstacles = obstacles
            
            self.D = 0.0
            self.psi = 0.0
            self.alpha = 0.0
            self.delta_alpha = 0.0
            
            self.ds = [0.0] * 9
            
            self.path = []
            
            self.dmin = min(self.ds)
            
            self._initialized = True
          
    def calculate_angle_to_north(self, x_current, y_current, x_target, y_target):
        dx = x_target - x_current
        dy = y_target - y_current
        
        theta = math.atan2(dy, dx)
        
        theta_degrees = math.degrees(theta)
        
        angle_to_north = 90 - theta_degrees
        
        if angle_to_north < 0:
            angle_to_north += 360
        
        return angle_to_north
    
    def calculate_distances_from_obstacles(self):
        obstacle_distances = []
        
        for i, obstacle in enumerate(self.obstacles):
            x, y, radius = obstacle
            
            d = radius - math.sqrt((self.position[0] - x) ** 2 + (self.position[1] - y) ** 2)
            
            obstacle_distances.append(d)
        
        obstacle_distances.sort()
        
        for i in range(9):
            self.ds[i] = obstacle_distances[i]
            
        self.dmin = min(self.ds)
          
    def calculate_distance(self, x1, y1, z1):
        return math.sqrt((self.target_position[0] - x1) ** 2 + (self.target_position[1] - y1) ** 2 + (self.target_position[2] - z1) ** 2)
            
    def get_state(self):
        return [self.position[0], self.position[1], self.D, self.psi, self.alpha, self.delta_alpha, *self.ds]
    
    def move_drone(self, action):
        actions = [-1, 0, 1]
        
        delta_psi = actions[action] * YAW_RATE * TIME_INTERVAL
        
        new_psi = math.radians(self.psi + delta_psi)
        
        delta_x = SPEED * math.sin(new_psi)
        delta_y = SPEED * math.cos(new_psi)
        
        self.position = (self.position[0] + delta_x, self.position[1] + delta_y, self.position[2])
        
        self.psi += delta_psi
        self.psi = self.psi % 360 
        
        self.D = self.calculate_distance(self.position[0], self.position[1], self.position[2])
        
        self.alpha = self.calculate_angle_to_north(self.position[0], self.position[1], self.target_position[0], self.target_position[1])
        
        self.delta_alpha = self.psi - self.alpha
        
        self.calculate_distances_from_obstacles()

        self.path.append(self.position)


class Environment:
    _instance = None
    _initialized = False
    
    def __new__(cls, *_, **__):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, x_max: float, y_max: float, z_max: float) -> None:
        if not self._initialized:
            self.x_max = x_max
            self.y_max = y_max
            self.z_max = z_max
            
            self.initialize_environment()
            
            self.agent = DroneMOAAPP((START_WAYPOINT[0], START_WAYPOINT[1], z_max), (x_max, y_max, z_max), self.obstacles)
            
            self._initialized = True
            
    def get_color(self, point_index):
        if point_index == 0:
            return 'go-'
        else:
            return 'ro-'
            
    def plot_points(self, points):
        for i in range(len(points) - 1):
            plt.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], [points[i][2], points[i + 1][2]], self.get_color(i))
    
    def plot_obstacles(self):
        for obstacle in self.obstacles:
            x, y, radius = obstacle
            h = self.y_max
            
            z = np.linspace(0, h, 100)
            theta = np.linspace(0, 2 * np.pi, 100)
            theta, z = np.meshgrid(theta, z)
            x = radius * np.cos(theta) + x
            y = radius * np.sin(theta) + y
            
            self.ax.plot_surface(x, y, z, alpha=0.5)
            
            self.ax.set_zlim([0, h])
            
    def show_environment(self):
        self.plot_points([self.start_point, *self.agent.path, self.target_point])
        self.plot_obstacles()
        plt.show()
    
    def initialize_environment(self):
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        
        self.ax.set_title("3D Environment")
        
        self.ax.set_xlim([0, self.x_max])
        self.ax.set_ylim([0, self.y_max])
        self.ax.set_zlim([0, self.z_max])
        
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        
        self.start_point = (START_WAYPOINT[0], START_WAYPOINT[1], self.z_max)
        self.target_point = (self.x_max, self.y_max, self.z_max)
        
        self.obstacles = []
        
        for _ in range(NUM_OBSTACLES):
            x = rnd.uniform(0, self.x_max)
            y = rnd.uniform(0, self.y_max)
            radius = rnd.uniform(0.3, 2)
            
            self.obstacles.append((x, y, radius))


if __name__ == "__main__":
    env = Environment(20, 20, 0.6)
    # print(f"Initial State: {env.agent.get_state()}")

    # # Target position
    # target_x, target_y, target_z = 5, 5, 0.6

    # # Move the drone in a curve
    # steps = 100  # Total steps to take
    # for i in range(steps):
    #     # Depending on the step, take left, right or straight actions
    #     if i % 10 < 5:  # Slight curve to the right
    #         env.agent.move_drone(2)  # Action 2 turns right
    #     else:  # Slight curve to the left
    #         env.agent.move_drone(0)  # Action 0 turns left

    #     # Print the drone's state every 10 steps to observe the curve
    #     if i % 10 == 0:
    #         print(f"Step {i}: {env.agent.get_state()}")
        
    #     # Check if we are close to the target position (simple stopping condition)
    #     x, y, z = env.agent.get_state()[:3]
    #     if abs(x - target_x) < 0.1 and abs(y - target_y) < 0.1:
    #         print(f"Drone reached close to the target at step {i}")
    #         break

    # # Final position
    # print(f"Final Position: {env.agent.get_state()}")
    
    for i in range(17):
        env.agent.move_drone(2) 
        
    for i in range(60):
        env.agent.move_drone(1)
        print(env.agent.get_state())
        
    # # for i in range(8):
    # #     env.agent.move_drone(0)
    
    # # for i in range(20):
    # #     env.agent.move_drone(1)
    
    env.show_environment()
