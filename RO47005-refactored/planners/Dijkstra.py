import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from random import choice as random_choice
import json
import copy
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from environment.variables import RAD_QUAD, RAD_OBST, RAD_NP, RAD_NBR, ENVIRONMENT, SPACING

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# DEFAULT_DRONES = DroneModel("cf2p")
# DEFAULT_NUM_DRONES = 1
# DEFAULT_PHYSICS = Physics("pyb")
# DEFAULT_GUI = True
# DEFAULT_RECORD_VISION = False
# DEFAULT_PLOT = True
# DEFAULT_USER_DEBUG_GUI = False
# DEFAULT_OBSTACLES = True
# DEFAULT_SIMULATION_FREQ_HZ = 240
# DEFAULT_CONTROL_FREQ_HZ = 48
# DEFAULT_DURATION_SEC = 20
# DEFAULT_OUTPUT_FOLDER = 'results'
# DEFAULT_COLAB = False
# plot_trajectory = True
# control_freq_hz=DEFAULT_CONTROL_FREQ_HZ
# num_drones=DEFAULT_NUM_DRONES

######### KDTREE #########
class kdtree:
    def __init__(self, _x, _y, _z, depth=0):
        self.pos = {'x': _x, 'y': _y, 'z': _z}
        self.left = None
        self.right = None
        self.depth = depth

    def findNodesinRange(self, min_cor, max_cor):
        '''Receives minimum and maximum x,y,z coordinates out of two points and returns all 
           obstacles that intersect this cube, since they potentially result in a collision
        '''
        if self is None:
            return []

        x_min, y_min, z_min = min_cor
        x_max, y_max, z_max = max_cor

        nodes_in_range = []

        # Checks if self (an obstacles) is inside cube of interests
        if ((x_min - RAD_OBST) <= self.pos['x'] <= (x_max + RAD_OBST) and
            (y_min - RAD_OBST) <= self.pos['y'] <= (y_max + RAD_OBST) and
            (z_min - RAD_OBST) <= self.pos['z'] <= (z_max + RAD_OBST)):
            nodes_in_range.append(self.getCoordinates())

        # In this KD-tree, depth=0 splits based on x-coordinate, depth=1 splits on y, depth=2 splits on z, this repeats
        axis = self.depth % 3
        key = 'xyz'[axis]

        # Decision: move left in tree and/or move right in tree
        if self.pos[key] + RAD_OBST >= min_cor[axis]:
            nodes_in_range.extend(self.left.findNodesinRange(min_cor, max_cor) if self.left else [])

        if self.pos[key] - RAD_OBST <= max_cor[axis]:
            nodes_in_range.extend(self.right.findNodesinRange(min_cor, max_cor) if self.right else [])

        return nodes_in_range

    def addNode(self, _x, _y, _z):
        '''Adds an obstacle to the KD-tree based on its x,y,z coordinate'''
        # Checks which coordinate to split on
        k = self.depth % 3
        if k == 0:
            key = 'x'
        elif k == 1:
            key = 'y'
        else:
            key = 'z'

        # Sort the obstacle in the trees
        if self.pos[key] >= (_x, _y, _z)[k]:
            if self.left is None:
                self.left = kdtree(_x, _y, _z, self.depth + 1)
            else:
                self.left.addNode(_x, _y, _z)
        else:
            if self.right is None:
                self.right = kdtree(_x, _y, _z, self.depth + 1)
            else:
                self.right.addNode(_x, _y, _z)

    def addNodeBatch(self, positions):
        '''Adds an array of obstacles'''
        for pos in positions:
            self.addNode(pos[0], pos[1], pos[2])

    def getCoordinates(self):
        '''Returns x,y,z coordinate of obstacle'''
        return [self.pos['x'],self.pos['y'],self.pos['z']]

def build_kdtree(points, depth=0):
    '''Builds the KD-tree such that it is more or less balanced'''
    if not points:
        return None

    # Check which coordinate to split on
    k = depth % 3

    # Sort points based on decision coordinate and set median as split condition
    points.sort(key=lambda point: point[k])
    median_index = len(points) // 2
    median_point = points[median_index]

    node = kdtree(*median_point, depth)

    node.left = build_kdtree(points[:median_index], depth + 1)
    node.right = build_kdtree(points[median_index + 1:], depth + 1)

    return node

def distPoint2Line(A,B,C):
    '''Given three points A,B,C, it determines whether point C is less than a distance
       RAD_OBST away from the line AB to check whether obstacle (sphere) with center C
       will intersect this line
    '''
    A, B, C = np.array(A), np.array(B), np.array(C)
    AB = B.T - A.T
    AC = C - A

    if AC.dot(AB) <= 0:
        return np.linalg.norm(AC) > RAD_OBST
    
    BC = C - B

    if BC.dot(AB) >= 0:
        return np.linalg.norm(BC) > RAD_OBST
    
    return (np.linalg.norm(np.cross(AB,AC)) / np.linalg.norm(AB))  > RAD_OBST



def create_3d_grid(x_range, y_range, z_range, spacing):
    # Create coordinate ranges
    x = np.arange(x_range[0], x_range[1], spacing)
    y = np.arange(y_range[0], y_range[1], spacing)
    z = np.arange(z_range[0], z_range[1], spacing)

    # Create 3D grid using meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Reshape X, Y, Z into a combined array
    XYZ = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    return XYZ

def euclideanDistance2(pointA, pointB):
    '''Returns euclidean distance between provided points'''
    return np.sqrt((pointA[0]-pointB[0])**2 + 
                   (pointA[1]-pointB[1])**2 + 
                   (pointA[2]-pointB[2])**2)

def generate_waypoints(waypoints, step_length):
    WAYPOINTS = []
    for i in range(len(waypoints) - 1):
        start = np.array(list(waypoints[i]))
        end = np.array(list(waypoints[i + 1]))

        num_steps = int(euclideanDistance2(start, end) / step_length)

        interpolated_points = [start + i * (end - start) / (num_steps + 1) for i in range(1, num_steps + 1)]

        WAYPOINTS.extend(interpolated_points)

    return np.array(WAYPOINTS)

def calcTotalPathLength(path):
    return sum([math.sqrt((path[i][0]-path[i-1][0])**2 + 
                    (path[i][1]-path[i-1][1])**2 + 
                    (path[i][2]-path[i-1][2])**2)
                    for i in range(1,len(path))])


class Dijkstra:
    def __init__(self, START_POINT, END_POINT, BOUNDS, SPHERES):
        X_BOUNDS, Y_BOUNDS, Z_BOUNDS = BOUNDS
        self.ROOT = build_kdtree(SPHERES)
        self.XYZ = create_3d_grid(X_BOUNDS, Y_BOUNDS, Z_BOUNDS, SPACING)
        self.filtered_XYZ = self.filter_grid(self.XYZ, X_BOUNDS, Y_BOUNDS, Z_BOUNDS)
        self.kdtree_points = cKDTree(self.filtered_XYZ)
        self.graph = {}
        self.buildGraph(self.graph)
        start_idx = self.kdtree_points.query(START_POINT, k=1)[1]
        self.start = self.filtered_XYZ[start_idx]
        end_idx = self.kdtree_points.query(END_POINT, k=1)[1]
        self.end = self.filtered_XYZ[end_idx]
        self.heap = {self.start: 0}
        for point in self.graph.keys():
            if point == self.start:
                continue
            self.heap[point] = np.Inf
    
        self.preds = {self.start: self.start}
        self.visited = set()

    def collisionCheck(self, pointA, pointB):
        '''Checks for collisions when moving along line segment from pointA to pointB'''
        # Find minimum and maximum x,y,z coordinates out of pointA and pointB
        min_cor = [min(pointA[i],pointB[i]) for i in range(3)]
        max_cor = [max(pointA[i],pointB[i]) for i in range(3)]

        # Find all obstacles that could potentially cause a collision
        potentialCollisionSpheres = self.ROOT.findNodesinRange(min_cor, max_cor)

        # Check for collisions out of all potential collision obstacles
        for sphere in potentialCollisionSpheres:
            if distPoint2Line(pointA, pointB, sphere) == False: # There is a collision 
                return False
        return True

    def filter_grid(self, XYZ, x_range, y_range, z_range):
        indices_to_remove = []
        for i, point in enumerate(XYZ):
            if (x_range[0] <= point[0] <= x_range[1]) and (y_range[0] <= point[1] <= y_range[1]) and (z_range[0] <= point[2] <= z_range[1]):
                if not self.collisionCheck(point, point):
                    indices_to_remove.append(i)
            else:
                indices_to_remove.append(i)

        filtered_XYZ = np.delete(XYZ, indices_to_remove, axis=0)

        res = [tuple(i) for i in filtered_XYZ]
        
        return res

    def findNeighbours(self, curr):
        nbrs = []
        dists = []
        max_dist = 1.05*np.sqrt(2)*SPACING
        min_dist =  0.1*SPACING

        indices = self.kdtree_points.query_ball_point(curr, max_dist)
        points_within_distance = [self.filtered_XYZ[i] for i in indices]

        for point in points_within_distance:
            dist = np.sqrt((curr[0] - point[0])**2 +
                            (curr[1] - point[1])**2 +
                            (curr[2] - point[2])**2)
            if dist <= max_dist and dist > min_dist:
                if self.collisionCheck(curr, point):
                    nbrs.append(point)
                    dists.append(dist)
        return nbrs, dists


    def buildGraph(self, graph):
        for i, point in enumerate(self.filtered_XYZ):
            if i%200==0: print('Iteration ', i)
            graph[point] = {}

            start_time = time.time()
            nbrs, dists = self.findNeighbours(point)
            end_time = time.time()
            # print('time: ', end_time - start_time)
            for i, nbr in enumerate(nbrs):
                graph[point][nbr] = dists[i]

    def solve(self):
        while self.heap:
            current = min(self.heap, key=self.heap.get)
            current_dist = self.heap.pop(current)
            # print('neighbor: ', neighbor)
            for neighbor, dist in self.graph[current].items():

                neighbor_dist = current_dist + dist

                if neighbor in self.visited:
                    continue

                if neighbor_dist < self.heap.get(neighbor, float('inf')):
                    self.heap[neighbor] = neighbor_dist
                    self.preds[neighbor] = current

            self.visited.add(current)

            if current == self.end:
                return self.preds

        print('Did not find a solution!')
        return self.preds # return None ?

    @property
    def path(self):
        point = self.end
        path = [self.end]
        while point != self.start:
            point = self.preds[point]
            path.append(point)
        return path[::-1]