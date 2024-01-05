import os

RAD_QUAD = 0.1                  # Radius of quadcopter
RAD_SPHERE = 0.2
RAD_OBST = RAD_SPHERE + RAD_QUAD       # Radius of obstacle, accounting for quadcopter size
RAD_NP = 1                      # Radius of new point
RAD_NBR = 1.5                   # Radius of neighbour
assert RAD_NP < RAD_NBR, "rad_np may never be greater than rad_nbr!"

current_file_directory = os.path.dirname(os.path.abspath(__file__))
load_maze = current_file_directory + '/maze.json'
load_walls = current_file_directory + '/walls.json'
ENVIRONMENT = load_walls