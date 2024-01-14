from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from random import random

def to_np_float_array(array: [np.ndarray, list]):
    """
    Make sure parameters are a numpy array
    """
    return array.astype(float) if type(array) is np.ndarray else np.array(array, float)

def normalize(vector):
    """
    convert vector to unit vector
    """
    return vector / np.linalg.norm(vector)

def spherical_to_cartesian(length, angle1, angle2):
    """
    Convert polar coordinates to Cartesian coordinates
    """
    x = length * np.cos(angle1) * np.cos(angle2)
    y = length * np.sin(angle1) * np.cos(angle2)
    z = length * np.sin(angle2)
    
    # Return the 2D vector as a NumPy array
    return np.array([x, y, z])

def angle_between_vectors(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (norm_a * norm_b)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    return angle

class Vehicle2d:
    """
    Generic Vehicle/Obstacle defined in 2D space.
    """
    def __init__(self, position: [np.ndarray, list, tuple], radius: float = 0.5, velocity: [np.ndarray, list, tuple] = np.zeros(3)):
        self.p = to_np_float_array(position)
        self.r = radius
        self.v = to_np_float_array(velocity)

class Agent2d(Vehicle2d):
    """
    Vehicle in 2d space that moves based on it's goal.
    """
    def __init__(self, position, radius = 1, goal: [np.ndarray, list, tuple] = np.zeros(3), max_speed=2.0, time_to_collision=1):
        super().__init__(position, radius)
        self.vmax = self.max_speed = max_speed
        self.goal = to_np_float_array(goal)
        self.v = self.desired_vel # Is updated before we move, so it's fine
        self.time_to_collision = time_to_collision

    @property
    def desired_vel(self):
        distance = self.goal - self.p
        norm = np.linalg.norm(distance)
        if norm < self.r:
            return np.zeros(3)
        direction = distance / norm
        return self.vmax * direction

    def forbiden_cone(self, obstacle: Vehicle2d) -> float:
        """
        we will just define a cone and check that the relative velocity is not in the cone instead of translating the cone and checking the sampled velocity
        """
        r = self.r + obstacle.r
        distance = obstacle.p - self.p

        # Find a vector perpendicular to distance
        if distance[0] == 0 and distance[1] == 0:
            # Handle special case for distance = [0, 0, z]
            normal = np.array([1, 0, 0])
        else:
            normal = np.array([-distance[1], distance[0], 0])
        
        # Project v onto the plane defined by normal
        projected_point = distance - np.dot(distance, normal) * normal

        # Distance from the projected point to the circle center
        projected_distance = np.linalg.norm(projected_point)
        
        # Calculate the angles of the tangent lines
        angle = np.arcsin(r / projected_distance)

        return angle

    # TODO bias toward goal
    # TODO subsaple along two dimentions
    def sample(self, n = 1_000, D2=False):
        samples = []
        for i in range(n):
            length = random() * self.max_speed
            alpha = random() * np.pi
            beta = random() * np.pi
            velocity = spherical_to_cartesian(length, alpha, beta)
            if D2:
                velocity[2] = 0 #for debug 2D
            samples.append(velocity)
        return samples

    def step(self, obstacles = [], max_samples = 1_000, debug = False, D2 = False):
        velocity_obstacles = [self.forbiden_cone(obstacle) for obstacle in obstacles]
            
        valid_samples = []   
        while len(valid_samples) < 5:
            valid_samples = []   
            counter = 0
            samples = self.sample(max_samples, D2)
            for sample in samples + [self.desired_vel]:
                for obstacle, max_angle in zip(obstacles, velocity_obstacles):
                    r = obstacle.r + self.r
                    distance = obstacle.p - self.p
                    versor = normalize(distance)
                    rel_vel = sample - obstacle.v
                    min_rel_vel = (distance - r * versor) / self.time_to_collision # Find the minimum velocity that can be on a collision line
                    if angle_between_vectors(rel_vel, distance) <= max_angle and min_rel_vel.dot(rel_vel - min_rel_vel) >= 0:
                        counter += 1
                        break
                else:
                    valid_samples.append(sample)
            print(f"invalid {counter}/{max_samples}")
    
        velocity = min(valid_samples, key=lambda speed: np.linalg.norm(self.desired_vel - speed))
        self.v = velocity
        if debug:
            return samples, valid_samples

    def speed_to_waypoint(self, t: float) -> None:
        """
        Function to give the PID the position we want to reach instead of working with velocities
        """
        return self.p + self.v * t

if __name__ == "__main__":
    import pygame
    FPS = 30
    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("")

    # Colors
    white = (255, 255, 255)
    blue = (0, 0, 255)
    green = (0, 255, 0)
    red = (255, 0, 0)
    black = (0, 0, 0)

    # Agent parameters
    agent_position = (800, 250, 0)
    agent_radius = 20
    agent_goal = (400, 400, 0)
    agent_max_speed = 100

    # Create Pygame clock object to control the frame rate
    clock = pygame.time.Clock()

    def rotate_2d_vector(vector, angle_rad):
        # Create the rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])

        # Rotate the vector using the matrix multiplication
        rotated_vector = np.dot(rotation_matrix, vector)

        return rotated_vector

    def draw_scene(agent, obstacles):
        screen.fill(white)

        # Draw agent
        pygame.draw.circle(screen, blue, agent.p.astype(int)[:2], agent_radius)

        # Draw obstacles
        for obstacle in obstacles:
            pygame.draw.circle(screen, red, obstacle.p.astype(int)[:2], int(obstacle.r))

        # Draw velocity obstacles
        start = agent.p[:2]
        for obstacle in obstacles:
            max_angle = agent.forbiden_cone(obstacle)
            #pygame.draw.circle(screen, blue, obstacle.p.astype(int)[:2], int(max_range))
            distance = (obstacle.p - agent.p)[:2]
            direction = distance / np.linalg.norm(distance)
            for sign in [-1, 1]:
                versor = rotate_2d_vector(direction, sign * max_angle)
                end = start + versor * 500
                pygame.draw.line(screen, black, start, end, 2)

        samples, valid_samples = agent.step(obstacles, 100, debug=True, D2=True)
        
        for sample in samples:
            pygame.draw.circle(screen, red, (agent.p + sample).astype(int)[:2], 2)
        
        for sample in valid_samples:
            pygame.draw.circle(screen, green, (agent.p + sample).astype(int)[:2], 2)

        agent.v = min(valid_samples, key=lambda speed: np.linalg.norm(agent.desired_vel - speed))
        pygame.display.flip()


    # Dummy obstacle data
    obstacle1_position = (500, 500, 0)
    obstacle1_goal = (500, 400, 0)
    obstacle_agent1 = Agent2d(obstacle1_position, radius=20, goal=obstacle1_goal, max_speed=45)

    obstacle2_position = (410, 390, 0)
    obstacle2_goal = np.array([100., 300, 0])
    obstacle_agent2 = Agent2d(obstacle2_position, radius=20, goal=obstacle2_goal, max_speed=0)

    obstacles = [obstacle_agent1, obstacle_agent2]

    agent = Agent2d(agent_position, agent_radius, agent_goal, agent_max_speed, time_to_collision=0.5)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        draw_scene(agent, obstacles)
        
        agent.p += agent.v * 1/FPS # Update agent position
        obstacle_agent1.p += obstacle_agent1.v * 1/FPS
        obstacle_agent2.p += obstacle_agent2.v * 1/FPS

        clock.tick(FPS)  # Set the frame rate

    pygame.quit()
