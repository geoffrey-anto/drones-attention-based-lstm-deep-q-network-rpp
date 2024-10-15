import matplotlib.pyplot as plt
from src.Config import YAW_RATE, TIME_INTERVAL, SPEED, NUM_OBSTACLES, START_WAYPOINT, \
    MINIMUM_SAFETY_DISTANCE, TARGET_ARRIVAL_THRESHOLD, OBSTACLES
import numpy as np
import math
import random as rnd


class DroneMOAAPP:

    def __init__(self, start_position, target_position, obstacles, train) -> None:
        self.position = start_position
        self.target_position = target_position
        
        self.obstacles = obstacles
        
        self.D = 0.0
        self.psi = 0.0
        self.alpha = 0.0
        self.delta_alpha = 0.0
        
        self.ds = [0.0] * 9
        
        self.path = []
        self.train = train
        self.path_count = 0
        
        self.dmin = min(self.ds)
          
    def calculate_angle_to_north(self, x_current, y_current, x_target, y_target):
        dx = x_target - x_current
        dy = y_target - y_current
        
        theta = math.atan2(dy, dx)
        
        theta_degrees = math.degrees(theta)
        
        angle_to_north = 90 - theta_degrees
        
        if angle_to_north < 0:
            angle_to_north += 360
        
        return math.radians(angle_to_north)
    
    def calculate_distances_from_obstacles(self):
        obstacle_distances = []
        
        for i, obstacle in enumerate(self.obstacles):
            x, y, radius = obstacle
            
            d = abs(math.sqrt((self.position[0] - x) ** 2 + (self.position[1] - y) ** 2) - radius)
            
            obstacle_distances.append(d)
        
        obstacle_distances.sort()
        
        for i in range(9):
            self.ds[i] = obstacle_distances[i]
            
        self.dmin = min(self.ds)
    
    def check_collision(self):
        return self.dmin <= MINIMUM_SAFETY_DISTANCE
          
    def calculate_distance(self, x1, y1, z1):
        return math.sqrt((self.target_position[0] - x1) ** 2 + (self.target_position[1] - y1) ** 2 + (self.target_position[2] - z1) ** 2)
            
    def get_state(self):
        return [self.position[0], self.position[1], self.D, self.psi, self.alpha, self.delta_alpha, *self.ds]
    
    def get_track_alignment(self):
        delta_x = self.target_position[0] - self.position[0]
        delta_y = self.target_position[1] - self.position[1]
        desired_heading = math.atan2(delta_y, delta_x)
        
        heading_diff = desired_heading - self.psi
        
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
        alignment_score = 1 - abs(heading_diff) / math.pi  # Normalize to [0, 1]
        
        return alignment_score
    
    def move_drone(self, action):
        actions = [-1, 0, 1]
        
        delta_psi = actions[action] * YAW_RATE * TIME_INTERVAL
        
        new_psi = self.psi + delta_psi
        
        delta_x = SPEED * math.sin(new_psi)
        delta_y = SPEED * math.cos(new_psi)
        
        self.position = (max(0, min(self.position[0] + delta_x, self.target_position[0])),
                 max(0, min(self.position[1] + delta_y, self.target_position[1])),
                 self.position[2])
        
        self.psi = new_psi
        self.psi = self.psi % 360
        
        self.D = self.calculate_distance(self.position[0], self.position[1], self.position[2])
        
        self.alpha = self.calculate_angle_to_north(self.position[0], self.position[1], self.target_position[0], self.target_position[1])
        
        self.delta_alpha = self.alpha - self.psi
        
        self.calculate_distances_from_obstacles()
        
        if not self.train:
            if self.path_count % 10 == 0:
                self.path.append(self.position)
            
            self.path_count += 1


class Environment: 

    def __init__(self, x_max: float, y_max: float, z_max: float, train=True) -> None:
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        
        self.initialize_environment()
        
        self.agent = DroneMOAAPP((START_WAYPOINT[0], START_WAYPOINT[1], z_max), (x_max, y_max, z_max), self.obstacles, train)
            
    def get_color(self, point_index):
        if point_index == 0:
            return 'g-'
        else:
            return 'r-'
        
    def plot_static_points(self):
        plt.plot([self.start_point[0]], [self.start_point[1]], [self.start_point[2]], 'go')
        plt.plot([self.target_point[0]], [self.target_point[1]], [self.target_point[2]], 'ro')
            
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
        self.plot_static_points()
        self.plot_points([*self.agent.path])
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
        
        if len(OBSTACLES) > 0:
            self.obstacles = OBSTACLES
        else:
            self.obstacles = []
            
            for _ in range(NUM_OBSTACLES):
                x = rnd.uniform(0, self.x_max)
                y = rnd.uniform(0, self.y_max)
                radius = rnd.uniform(0.1, 0.4)
                
                self.obstacles.append((x, y, radius))
        
        print(self.obstacles)
    
    def step(self, action):
        self.agent.move_drone(action)  # Move the drone based on the action
        
        state = self.agent.get_state()  # Get the current state of the drone
        reward = 0  # Initialize reward
        
        if self.agent.check_collision():
            reward = -10  # Collision reward
        elif self.agent.D < TARGET_ARRIVAL_THRESHOLD:  # Assuming TARGET_ARRIVAL_THRESHOLD is the arrival condition
            reward = 20  # Arrival reward
        else:
            current_state = self.agent.position
            
            if current_state[0] < 0 or current_state[0] > self.x_max or current_state[1] < 0 or current_state[1] > self.y_max:
                reward = -5
            
            # Calculate distance-based reward
            distance_reward = 5 * (0.8 * self.agent.dmin / self.agent.D)
            
            # Track angle reward (assuming you have a function to calculate how well the drone is aligned with its track)
            track_angle_reward = 5 * self.agent.get_track_alignment()
            
            # Sum up the distance and track angle rewards
            reward = distance_reward + track_angle_reward
        
        return state, reward, self.agent.D <= TARGET_ARRIVAL_THRESHOLD
