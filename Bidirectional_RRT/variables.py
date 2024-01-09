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
load_city = current_file_directory + '/city.json'

load_walls_env = current_file_directory + '/spheres_walls.urdf'
load_maze_env = current_file_directory + '/spheres_maze.urdf'
load_city_env = current_file_directory + '/spheres_city.urdf'

load_random_maze = current_file_directory + '/random_maze.json'
load_random_walls = current_file_directory + '/random_walls.json'
load_random_city = current_file_directory + '/random_city.json'
load_random_walls_env = current_file_directory + '/spheres_random_walls.urdf'
load_random_maze_env = current_file_directory + '/spheres_random_maze.urdf'
load_random_city_env = current_file_directory + '/spheres_random_city.urdf'

ENVIRONMENT = load_random_walls
ENVIRONMENT_SPHERES = load_random_walls_env