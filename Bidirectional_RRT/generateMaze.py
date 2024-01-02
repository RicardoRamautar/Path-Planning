from mazelib import Maze
from mazelib.generate.Prims import Prims
import json
import random
import numpy as np
import time
import json
import os
from variables import RAD_QUAD, RAD_SPHERE

HEIGHT = 4
SIZE = 15
# RAD_QUAD = 0.1
# RAD_SPHERE = 0.15
SPHERE_DIAMETER = 2 * RAD_SPHERE  # Adjust this value to change the radius of the spheres

if (np.sqrt(2)-1)*RAD_SPHERE > RAD_QUAD:
    print("WARNING: Possibility of holes in walls")

def euclideanDistance(pointA, pointB):
    return np.sqrt((pointA[0]-pointB[0])**2 + 
                   (pointA[1]-pointB[1])**2 + 
                   (pointA[2]-pointB[2])**2)

def collisioncheck(maze, point):
    for obstacle in maze:
        if euclideanDistance(obstacle, point) < (SPHERE_DIAMETER/2 + RAD_QUAD*1.2):
            return False
    return True

def randomPoint():
    x = round(random.uniform(0.1, (SIZE - 1)*SPHERE_DIAMETER),2)
    y = round(random.uniform(0.1, (SIZE - 1)*SPHERE_DIAMETER),2)
    z = round(random.uniform(0.1, (HEIGHT - 1)*SPHERE_DIAMETER),2)
    point = (x,y,z)

    while not collisioncheck(maze_layout["obstacles"], point):
        x = round(random.uniform(0.1, (SIZE - 1)*SPHERE_DIAMETER),2)
        y = round(random.uniform(0.1, (SIZE - 1)*SPHERE_DIAMETER),2)
        z = round(random.uniform(RAD_SPHERE*1.5, (HEIGHT - 1)*SPHERE_DIAMETER),2)
        point = (x,y,z)

    return point

def distPoint2Line(A,B,C):
    A = np.array(list(A))
    B = np.array(list(B))
    C = np.array(list(C))
    AB = B.T - A.T
    AC = C - A

    if AC.dot(AB) <= 0:
        return np.linalg.norm(AC) > RAD_SPHERE
    
    BC = C - B

    if BC.dot(AB) >= 0:
        return np.linalg.norm(BC) > RAD_SPHERE
    
    return (np.linalg.norm(np.cross(AB,AC)) / np.linalg.norm(AB))  > RAD_SPHERE

m = Maze()
m.generator = Prims(SIZE // 2, SIZE // 2)
m.generate()

maze_layout = {"obstacles": []}

x_min, y_min = np.Inf, np.Inf
x_max, y_max = 0, 0

for y, line in enumerate(m.grid):
    if y < y_min: y_min = y
    elif y > y_max:  y_max = y

    for x, is_wall in enumerate(line):
        if x < x_min: x_min = x
        elif x > x_max: x_max = x

        if is_wall:
            for z in range(HEIGHT):
                maze_layout["obstacles"].append([x * SPHERE_DIAMETER, y * SPHERE_DIAMETER, z * SPHERE_DIAMETER + SPHERE_DIAMETER/2])

got_points = False
while not got_points:
    start = randomPoint()
    time.sleep(0.2)
    end = randomPoint()

    intersection = True
    for obstacle in maze_layout["obstacles"]:
        if not distPoint2Line(start, end, obstacle):
            got_points = True

xbound = [0.1, (SIZE - 1)*SPHERE_DIAMETER]
ybound = [0.1, (SIZE - 1)*SPHERE_DIAMETER]
zbound = [RAD_SPHERE*1.5, (HEIGHT - 1)*SPHERE_DIAMETER]

maze_layout["start"] = start
maze_layout["end"] = end
maze_layout["xbounds"] = xbound
maze_layout["ybounds"] = ybound
maze_layout["zbounds"] = zbound

current_file_directory = os.path.dirname(os.path.abspath(__file__))
print(current_file_directory)

# path = os.path.join(current_file_directory, "/maze.json")
path = current_file_directory + "/maze.json"

with open(path, "w") as fp:
    json.dump(maze_layout, fp)
