import numpy as np
import matplotlib.pyplot as plt
import json
from variables import RAD_OBST
import os
from create_urdf import create_urdf_file

spacing = 1.5*RAD_OBST

def create_mesh_grid(n, R):
    x = np.arange(0, n * R, R)
    y = np.arange(0, n * R, R)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((X.shape[0], X.shape[1]))  # You can modify this based on your specific formula
    
    points = [(X[i, j], Y[i, j], Z[i, j]) for i in range(X.shape[0]) for j in range(X.shape[1])]
    return points

def assign_z_values(points, R, z_range):
    new_points = []
    for point in points:
        x, y, _ = point
        height = np.random.randint(z_range[0], z_range[1] + 1)
        for j in range(1, height + 1):
            new_points.append((x, y, j * 0.5 * R))
    return points + new_points

def assign_z_values_center(points, R, z_range):
    new_points = []
    for point in points:
        x, y, _ = point
        height = z_range[1]
        for j in range(1, height + 1):
            new_points.append((x, y, j * 0.5 * R))
    return points + new_points

def plot_mesh_points(points):
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', marker='o', label='Points')
    plt.colorbar(label='Values')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Mesh Grid - Points')
    plt.legend()
    plt.show()

def find_min_max_coordinates(points):
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    return min_x, max_x, min_y, max_y

n = 20  # Change this to your desired grid size
z_range = [5,40]

points = create_mesh_grid(n, spacing)
new_points = assign_z_values(points, RAD_OBST, z_range)

min_x, max_x, min_y, max_y = find_min_max_coordinates(new_points)

center_points = []
for point in new_points:
    x,y,_ = point
    distance_to_center = np.sqrt((x - max_x/2)**2 + (y - max_y/2)**2)
    if distance_to_center < 2 * spacing:
        center_points.append(point)
            
center_points = assign_z_values_center(center_points, RAD_OBST, z_range)

new_points = center_points + new_points

for i in range(len(new_points)):
    x,y,z = new_points[i]
    if (x<4*spacing and y<4*spacing) or (x>max_x - 4*spacing and y>max_y - 4*spacing):
        new_points[i] = (new_points[i][0], new_points[i][1], 0)

SPHERES = new_points
START_POINT = (0,0,z_range[1]*RAD_OBST/4)
END_POINT = (max_x, max_y, z_range[1]*RAD_OBST/4)
X_BOUNDS = [min_x, max_x]
Y_BOUNDS = [min_y, max_y]
Z_BOUNDS = [z_range[0]*RAD_OBST/2, z_range[1]*0.7*RAD_OBST/2]

data = {'obstacles': new_points,
        'start': START_POINT,
        'end': END_POINT,
        'xbounds': X_BOUNDS,
        'ybounds': Y_BOUNDS,
        'zbounds': Z_BOUNDS}

current_file_directory = os.path.dirname(os.path.abspath(__file__))
path = current_file_directory + "/random_city.json"
spheres_path = current_file_directory + "/spheres_random_city.urdf"

create_urdf_file(new_points, spheres_path)

with open(path, 'w') as file:
    json.dump(data, file)
