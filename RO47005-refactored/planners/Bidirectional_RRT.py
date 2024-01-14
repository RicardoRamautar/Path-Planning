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
from scipy.interpolate import splprep, splev
from environment.variables import RAD_OBST, RAD_NP, RAD_NBR, MAX_DIST_TREE_CONNECT, RRT_ITERATIONS

######### KD-tree #########
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

######### PLOT FUNCTION #########
def plotTrees(tree, path):
    plt.figure()
    ax = plt.gca()
    for sphere in self.SPHERES:
        circle = plt.Circle((sphere[0], sphere[1]), RAD_OBST, color='b')
        ax.add_patch(circle)

    for c, v in tree.items():
        pred = v.predecessor

        if pred == None:
            continue

        c_pred = pred.getNpCoordinates()
        plt.plot([c[0], c_pred[0]], [c[1], c_pred[1]], color='c') 
        plt.scatter(c[0], c[1], color='c')   
        
    x_coords = [point[0] for point in path]
    y_coords = [point[1] for point in path]
    ax.plot(x_coords, y_coords, marker='o', color='r')

    plt.scatter(START_POINT[0], START_POINT[1], color='g', s=80)
    plt.scatter(END_POINT[0], END_POINT[1], color='g', s=80)

    plt.show()


######### LOGIC #########
class Vertex:
    def __init__(self, position, pred=None, dist=0):
        self.x, self.y, self.z = position
        self.predecessor = pred
        self.min_path_length = dist        # Distance along path from source to vertex
        # self.successor = None

    def getNpCoordinates(self):
        return np.array([self.x, self.y, self.z])
    
    def getTupleCoordinates(self):
        return (self.x, self.y, self.z)
    
def euclideanDistance(pointA, pointB):
    #return np.linalg.norm(pointA - pointB) No because you used tuples...
    return np.sqrt((pointA[0]-pointB[0])**2 + 
                   (pointA[1]-pointB[1])**2 + 
                   (pointA[2]-pointB[2])**2)


def distPoint2Line(A,B,C):
    '''Given three points A,B,C, it determines whether point C is less than a distance
       RAD_OBST away from the line AB to check whether obstacle (sphere) with center C
       will intersect this line
    '''
    AB = B.T - A.T
    AC = C - A

    if AC.dot(AB) <= 0:
        return np.linalg.norm(AC) > RAD_OBST
    
    BC = C - B

    if BC.dot(AB) >= 0:
        return np.linalg.norm(BC) > RAD_OBST
    
    return (np.linalg.norm(np.cross(AB,AC)) / np.linalg.norm(AB))  > RAD_OBST

class Bidirectional_RRT():
    def __init__(self, START_POINT, END_POINT, BOUNDS, SPHERES):
        self.ROOT = build_kdtree(SPHERES)
        self.SPHERES = SPHERES
        self.X_BOUNDS, self.Y_BOUNDS, self.Z_BOUNDS = BOUNDS
        self.trees = {
            'start' : {START_POINT: Vertex(START_POINT)},
            'end' : {END_POINT: Vertex(END_POINT)},
            'connections' : []
        }
        self.START_POINT = START_POINT
        self.END_POINT = END_POINT

    def findNearestNode(self, random_point, tree):
        '''Finds nearest node in tree to the random point''' 
        nearest_node = None
        min_dist = np.Inf

        #nearest_node = min(self.tree.values(), key = lambda node_pos, node : np.linalg.norm(node_pos - random_point))[1]
        for node_pos, node in tree.items():
            dist = euclideanDistance(node_pos, random_point)

            if dist < min_dist:
                nearest_node = node
                min_dist = dist
            
        return nearest_node

    def findNewPoint(self, random_point, nearest_node):
        '''Finds the point on the line connecting the random point and nearest node
           a distance RAD_NP away from the nearest node
        ''' 
        pos_nearest_node = nearest_node.getNpCoordinates()

        dist = euclideanDistance(random_point, pos_nearest_node)

        # if dist=0, random_point == nearest_node.pos -> return to get new random point
        if dist == 0: return None
        else: t = RAD_NP / dist

        # if dist random point to nearest node is less than RAD_NP, set new point to random point
        if (t < 0) or (t > 1):
            new_point = random_point
        else:
            new_point = pos_nearest_node.T + t*(random_point.T - pos_nearest_node.T)
        
        return new_point

    def collisionCheck(self, pointA, pointB):
        '''Checks for collisions when moving along line segment from pointA to pointB'''
        # Find minimum and maximum x,y,z coordinates out of pointA and pointB
        min_cor = [min(pointA[i],pointB[i]) for i in range(3)]
        max_cor = [max(pointA[i],pointB[i]) for i in range(3)]

        # Find all obstacles that could potentially cause a collision
        potentialCollisionSpheres = self.ROOT.findNodesinRange(min_cor, max_cor)

        # Check for collisions out of all potential collision obstacles
        for sphere in potentialCollisionSpheres:
            if distPoint2Line(pointA, pointB, sphere) == False:
                return False
        return True

    def addNewPoint(self, new_point, tree):
        '''Adds a new point to  the tree and returns all neighbours of the new point'''
        neighbours = {}
        min_path_length = np.Inf
        predecessor = None

        # Iterate through all nodes in the tree
        for node in tree.values():
            node_pos = node.getNpCoordinates()

            # Check if node can be a neighbour of the new point
            edge_length = euclideanDistance(new_point, node_pos)
            if edge_length >= RAD_NBR:
                continue

            # Check if connection between node and new point is possible
            if not self.collisionCheck(node_pos, new_point):
                continue
            
            path_length = node.min_path_length + edge_length

            if edge_length < RAD_NBR:
                neighbours[node] = edge_length

                if path_length < min_path_length:
                    min_path_length = path_length
                    predecessor = node

        if not neighbours:
            return None, None

        # Add new point to the tree and connect it to the node resulting in the smallest path to the source out of all neighbours
        new_node = Vertex(new_point, predecessor, min_path_length)
        tree[tuple(new_point)] = new_node

        return neighbours, new_node

    def updateNeighbours(self, neighbours, new_node, tree):
        for nbr_node, edge_length in neighbours.items():
            nbr_position = nbr_node.getTupleCoordinates()
            if (new_node.min_path_length + edge_length) < nbr_node.min_path_length:
                tree[nbr_position].predecessor = new_node
                tree[nbr_position].min_path_length = new_node.min_path_length + edge_length
                        
    def returnPartialPath(self, node, end, depth=0):
        if node.getTupleCoordinates() == end:
            return [end]
        
        path = self.returnPartialPath(node.predecessor, end, depth+1)
        path.append(node.getTupleCoordinates())

        return path
        
    def getBestConnection(self, trees):
        '''Finds connection between the two trees resulting in the shortest path
           from start to end
        '''
        assert trees['connections'], "No connection possible!"

        best_connection = None
        minPathLength = np.Inf
        for node_s, node_e, edgeLength in trees['connections']:
            path_length = node_s.min_path_length + node_e.min_path_length + edgeLength
            if path_length< minPathLength:
                best_connection = [node_s, node_e]
                minPathLength = path_length
        return best_connection
    
    @property
    def path(self):
        '''Finds shortest path given the two trees'''
        best_connection = self.getBestConnection(self.trees)
        start_path = self.returnPartialPath(best_connection[0], self.START_POINT)
        end_path = self.returnPartialPath(best_connection[1], self.END_POINT)
        combined_list = start_path + list(reversed(end_path))
        return combined_list

    def solve(self, N = RRT_ITERATIONS):
        '''Implements the bidirectional RRT* algorithm'''
        # Store tree with start as origin and tree with end as origin, plus connections between the two trees

        total_iterations = 500
        num_segments = 20
        segment_length = total_iterations // num_segments

        for i in range(N):
            if i%50==0:
                print("Iteration: ", i)

            # Switch between trees every iteration (alternate expansion)
            if i%2 : 
                current_tree = 'start'
                # Implements directionality by sometimes taking random point equal to a node from the opposite tree
                if i%19==0:
                    random_point = random.choice(list(self.trees['end'].values())).getNpCoordinates()
                else:
                    x = round(random.uniform(self.X_BOUNDS[0], self.X_BOUNDS[1]),1)
                    y = round(random.uniform(self.Y_BOUNDS[0], self.Y_BOUNDS[1]),1)
                    z = round(random.uniform(self.Z_BOUNDS[0], self.Z_BOUNDS[1]),1)
                    random_point = np.array([x,y,z])
            else: 
                current_tree = 'end'
                # Implements directionality by sometimes taking random point equal to a node from the opposite tree
                if i%20==0:
                    random_point = random.choice(list(self.trees['start'].values())).getNpCoordinates()
                else:
                    x = round(random.uniform(self.X_BOUNDS[0], self.X_BOUNDS[1]),1)
                    y = round(random.uniform(self.Y_BOUNDS[0], self.Y_BOUNDS[1]),1)
                    z = round(random.uniform(self.Z_BOUNDS[0], self.Z_BOUNDS[1]),1)
                    random_point = np.array([x,y,z])

            # Find nearest node to random point in current tree (Nearest)
            nearest_node = self.findNearestNode(random_point, self.trees[current_tree])

            # Find new point for current tree (Steer)
            new_point = self.findNewPoint(random_point, nearest_node)
            if new_point is None:
                continue

            # Add new_point to tree and find its neighbours
            neighbours, new_node = self.addNewPoint(new_point, self.trees[current_tree])
                    
            if not neighbours:
                continue

            self.updateNeighbours(neighbours, new_node, self.trees[current_tree])

        
        self.connectTrees()

        return self.trees


    def connectTrees(self):
        '''Finds connections between the two trees'''
        for node_pos_s, node_s in self.trees['start'].items():
            for node_pos_e, node_e in self.trees['end'].items():
                dist = euclideanDistance(node_pos_e, node_pos_s)

                if dist < MAX_DIST_TREE_CONNECT:
                    if self.collisionCheck(node_s.getNpCoordinates(), node_e.getNpCoordinates()):
                        self.trees['connections'].append([node_s, node_e, dist])


def generate_waypoints(waypoints, step_length):
    WAYPOINTS = []
    for i in range(len(waypoints) - 1):
        start = np.array(list(waypoints[i]))
        end = np.array(list(waypoints[i + 1]))

        num_steps = int(euclideanDistance(start, end) / step_length)

        interpolated_points = [start + i * (end - start) / (num_steps + 1) for i in range(1, num_steps + 1)]

        WAYPOINTS.extend(interpolated_points)

    return np.array(WAYPOINTS)

def calcTotalPathLength(path):
    return sum([math.sqrt((path[i][0]-path[i-1][0])**2 + 
                    (path[i][1]-path[i-1][1])**2 + 
                    (path[i][2]-path[i-1][2])**2)
                    for i in range(1,len(path))])