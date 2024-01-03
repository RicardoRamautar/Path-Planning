import random
import time
import json
import os
# from variables import RAD_SPHERE
RAD_SPHERE = 0.12

def createWall(width, height, radius, y):
    time.sleep(0.1)
    # Choose a random starting point, ensuring there's space for a square
    random_start_x = random.randint(0, width - 3)
    random_start_y = random.randint(0, height - 3)

    positions = []
    removed_positions = []
    for i in range(width):
        for j in range(height):
            # Check if the current sphere is part of the square to be removed
            if random_start_x <= i <= random_start_x + 1 and random_start_y <= j <= random_start_y + 1:
                removed_positions.append([i * 2 * radius, y, j * 2 * radius])
                continue
            positions.append([i * 2 * radius, y, j * 2 * radius])

    return positions, removed_positions

def createWallSequence(nr, width, height, radius, spacing_walls):
    walls = []
    removed_positions = []
    for l in range(nr):
        positions, removed = createWall(width, height, radius, l * spacing_walls)
        walls.extend(positions)
        removed_positions.extend(removed)
    return walls, removed_positions

# Number of horizontal spheres
width = 10

# Number of vertical spheres
height = 10

# Number of consecutive walls
nr = 3

# spacing between consecutive walls
spacing_walls = 4

# Radius of the spheres -> needs to be equal to RAD_OBST
radius = RAD_SPHERE

# Generate sphere positions and removed positions
sphere_positions, removed_positions = createWallSequence(nr, width, height, radius, spacing_walls)

start_position = [removed_positions[0][0], removed_positions[0][1] - 1, removed_positions[0][2]]
end_position = [removed_positions[-1][0], removed_positions[-1][1] + 1, removed_positions[-1][2]]

x_bounds = [0,width*2*radius-radius]
y_bounds = [-2,nr*spacing_walls]
z_bounds = [0,height*2*radius-radius]

data = {'obstacles': sphere_positions,
        'start': start_position,
        'end': end_position,
        'xbounds': x_bounds,
        'ybounds': y_bounds,
        'zbounds': z_bounds}

script_dir = os.path.dirname(__file__)
filename = os.path.join(script_dir, 'walls.json')

with open(filename, 'w') as file:
    json.dump(data, file)

