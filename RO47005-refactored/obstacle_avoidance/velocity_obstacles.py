from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from random import randrange

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

def polar_to_cartesian(length, angle):
    """
    Convert polar coordinates to Cartesian coordinates
    """
    x = length * np.cos(angle)
    y = length * np.sin(angle)
    
    # Return the 2D vector as a NumPy array
    return np.array([x, y])

class Line:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def intersection(self, line):
        x = (line.b - self.b) / (self.a - line.a)
        return np.array([x, self(x)], float)

    def intersection_circle(self, center, r):
        h, k = center
    
        # Coefficients for the quadratic equation
        a = 1 + self.a**2
        b = 2 * (self.a * (self.b - k) - h)
        c = h**2 + (self.b - k)**2 - r**2
    
        # Solve the quadratic equation
        discriminant = b**2 - 4 * a * c
        assert discriminant > 0, discriminant
        x = (-b + np.sqrt(discriminant)) / (2 * a)

        return np.array([x, self(x)])

    @property
    def versor(self):
        """
        Extract direction versor
        """
        return normalize(np.array([0, self(0)]) - np.array([1, self(1)]))
    

    def __call__(self, *x):
        x = to_np_float_array(x)
        return self.a * x + self.b

    def __add__(self, vector: np.ndarray):
        """
        perform the translation of the line by the given vector
        """
        self.b += vector[1] - self.a * vector[0]
        return self

    def __sub__(self, vector: np.ndarray):
        return self + (- vector)

    def __iter__(self):
        return iter([self.a, self.b])
    
class DirectedLine(Line):
    """
    Half of a plane defined by a line and a point in that section
    """
    def __init__(self, a, b, point):
        super().__init__(a, b)
        self.direction = self.sign(point)

    def sign(self, point):
        x, y = point
        return (y - self(x) > 1) * 2 - 1

    def __contains__(self, point):
        return True if self.direction == self.sign(point) else False
        #return self.direction * self.sign(point)

def tangent_lines(point: np.ndarray, circle_center: np.ndarray, circle_radius: float):
    """
    Find two tangent lines to a given circle
    """
    x0, y0 = point
    cx, cy = circle_center

    # Calculate the distance between the point and the circle center
    d = np.sqrt((x0 - cx)**2 + (y0 - cy)**2)

    # Calculate the angle between the line connecting the point and circle center and the x-axis
    theta = np.arctan2(y0 - cy, x0 - cx)

    # Calculate the angles of the tangent lines
    alpha = np.arcsin(circle_radius / d)

    # Calculate the angles of the tangent lines
    angle1 = theta + alpha
    angle2 = theta - alpha

    # Calculate the coordinates of the tangent points
    x1 = cx + circle_radius * np.cos(angle1)
    y1 = cy + circle_radius * np.sin(angle1)

    x2 = cx + circle_radius * np.cos(angle2)
    y2 = cy + circle_radius * np.sin(angle2)

    # Calculate the equations of the tangent lines
    slope1 = np.tan(angle1)
    intercept1 = y0 - slope1 * x0

    slope2 = np.tan(angle2)
    intercept2 = y0 - slope2 * x0

    return Line(slope1, intercept1), Line(slope2, intercept2)

def perpendicular_bisector(line, point1, point2, ration = 0.5):
    bisector = Line(-1 / line.a, 0)
    bisector += point1 + (point2 - point1) * ration 
    
    return bisector

def inside_area(point: np.ndarray, lines: list[DirectedLine]):
    """
    Check if a point is in an area delimited by a list of lines
    For each line we check that our point is in the same section as the obstacle
    """
    for line in lines:
        if not (point in line):
            return False
    return True

class Vehicle2d:
    """
    Generic Vehicle/Obstacle defined in 2D space.
    """
    def __init__(self, position: [np.ndarray, list, tuple], radius: float = 0.5, velocity: [np.ndarray, list, tuple] = np.zeros(2, float)):
        self.p = to_np_float_array(position)
        self.r = radius
        self.v = to_np_float_array(velocity)

class Agent2d(Vehicle2d):
    """
    Vehicle in 2d space that moves based on it's goal.
    """
    def __init__(self, position, radius = 1, goal: [np.ndarray, list, tuple] = np.array([5.0, 5.0]), max_speed=2.0, time_to_collision=1):
        super().__init__(position, radius)
        self.vmax = self.max_speed = max_speed
        self.goal = to_np_float_array(goal)
        self.v = self.desired_vel # Is updated before we move, so it's fine
        self.time_to_collision = time_to_collision

    @property
    def desired_vel(self):
        """
        Find ideal velocity to reach the goal
        """
        distance = self.goal - self.p
        norm = np.linalg.norm(distance)
        if norm < self.r:
            return np.zeros(2)
        direction = distance / norm
        return self.vmax * direction

    def forbiden_region(self, obstacle: Vehicle2d):
        """
        return two lines that delimit the velocity region where we may have a colision.
        Obtained by taking the tangent lines to the obstacle and then shifting the reagion based on the relative velocity 
        """
        r = self.r + obstacle.r
        distance = obstacle.p - self.p
        tangent_1, tangent_2 = tangent_lines(self.p, obstacle.p, r)
        slow_limit = perpendicular_bisector(Line(distance[1] / distance[0], 0), self.p, obstacle.p, ration=1/self.time_to_collision) - (normalize(distance) * r / self.time_to_collision)
        tangent_1, tangent_2, slow_limit = (DirectedLine(*line, obstacle.p) for line in [tangent_1, tangent_2, slow_limit])
        shifted_tangent_1, shifted_tangent_2, shifted_slow_limit = (line + obstacle.v for line in [tangent_1, tangent_2, slow_limit])
        return shifted_tangent_1, shifted_tangent_2, shifted_slow_limit

    def sample(self, n = 20_000) -> list[np.ndarray]:
        """
        Sample n random velocities around the object smaller than max_speed
        """
        samples = []
        for i in range(n):
            length = randrange(0, self.max_speed * 100, self.max_speed) / 100
            direction = randrange(0, 360)
            velocity = polar_to_cartesian(length, direction)
            samples.append(velocity)
        return samples

    def step(self, *, obstacles: list[Agent2d] = [], max_samples = 20_000) -> None:
        """
        Calculate next velocity based on the goal given a list of obstacles 
        """
        velocity_obstacles = [self.forbiden_region(obstacle) for obstacle in obstacles]
            
        valid_samples = [np.array([0, 0])]   
        for sample in self.sample(max_samples):
            if not any(inside_area(self.p + sample, velocity_obstacle) for velocity_obstacle in velocity_obstacles):
                valid_samples.append(sample)
    
        self.v = min(valid_samples, key=lambda speed: np.linalg.norm(self.desired_vel - speed))

    def speed_to_waypoint(self, t: float) -> None:
        """
        Function to give the PID the position we want to reach instead of working with velocities
        """
        return self.p + self.v * t

def test_samples_Angelo():
    agent_position = [0.0, 0.0]
    agent_radius = 1.0
    agent_max_speed = 2.0
    sampled_vel = np.array([1, 3])
    agent = Agent2d(agent_position, agent_radius, agent_max_speed)

    # Dummy obstacle data
    obstacle_position1 = [5.4, 1.0]
    obstacle_agent1 = Agent2d(obstacle_position1)

    obstacles = [obstacle_agent1]

    velocity_obstacles = (agent.forbiden_region(obstacle) for obstacle in obstacles)

    plt.figure(figsize=(8, 8))

    agent_circle = plt.Circle(agent.p, agent.r, color='blue', alpha=0.5, label='Agent')
    plt.gca().add_patch(agent_circle)

    arrow_agent = FancyArrowPatch(agent.p, agent.p + sampled_vel,
                                      color='blue', arrowstyle='->', mutation_scale=15, label='Agent Velocity')
    plt.gca().add_patch(arrow_agent)

    for obstacle, velocity_obstacle in zip(obstacles, velocity_obstacles):
        t1, t2, slimit = velocity_obstacle
        distance = obstacle.p - agent.p
        for tangent in [t1, t2]:
            start = slimit.intersection(tangent)
            end = (start + 3*(distance[0]/normalize(distance)[0]) * ( - np.sign(distance[0])) * tangent.versor)
            plt.plot((start[0], end[0]), (start[1], end[1]))
            
        start = slimit.intersection(t1)
        end = slimit.intersection(t2)
        plt.plot((start[0], end[0]), (start[1], end[1]))

        agent_circle = plt.Circle(obstacle.p, obstacle.r, color='black', alpha=0.5, label='Obstacle')
        plt.gca().add_patch(agent_circle)

    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Run the test
    test_samples_Angelo()
