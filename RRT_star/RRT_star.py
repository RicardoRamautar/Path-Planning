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
from variables import RAD_QUAD, RAD_OBST, RAD_NP, RAD_NBR, ENVIRONMENT

from CtrlRRT import CtrlRRT
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
plot_trajectory = True

######### LOAD MAP #########
f = open(ENVIRONMENT)
# f = open(load_maze)
data = json.load(f)
SPHERES = data['obstacles']
start_point = data['start']
end_point = data['end']

######### GLOBAL VARIABLES #########
START_POINT = tuple(start_point)
END_POINT = tuple(end_point)

N = 700                     # Number of iterations
X_BOUNDS = data['xbounds']
Y_BOUNDS = data['ybounds']
Z_BOUNDS = data['zbounds']

# RAD_QUAD = 0.25                 # Radius of quadcopter
# RAD_OBST = 0.5 + RAD_QUAD       # Radius of obstacle, accounting for quadcopter size
# RAD_NP = 1                      # Radius of new point
# RAD_NBR = 1.5                   # Radius of neighbour
assert RAD_NP < RAD_NBR, "rad_np may never be greater than rad_nbr!"

######### KD-tree #########
class kdtree:
    def __init__(self, _x, _y, _z, depth=0):
        self.pos = {'x': _x, 'y': _y, 'z': _z}
        self.left = None
        self.right = None
        self.depth = depth

    def findNodesinRange(self, min_cor, max_cor):
        if self is None:
            return []

        x_min, y_min, z_min = min_cor
        x_max, y_max, z_max = max_cor

        nodes_in_range = []

        if ((x_min - RAD_OBST) <= self.pos['x'] <= (x_max + RAD_OBST) and
            (y_min - RAD_OBST) <= self.pos['y'] <= (y_max + RAD_OBST) and
            (z_min - RAD_OBST) <= self.pos['z'] <= (z_max + RAD_OBST)):
            nodes_in_range.append(self.getCoordinates())

        axis = self.depth % 3
        key = 'xyz'[axis]

        if self.pos[key] + RAD_OBST >= min_cor[axis]:
            nodes_in_range.extend(self.left.findNodesinRange(min_cor, max_cor) if self.left else [])

        if self.pos[key] - RAD_OBST <= max_cor[axis]:
            nodes_in_range.extend(self.right.findNodesinRange(min_cor, max_cor) if self.right else [])

        return nodes_in_range

    def addNode(self, _x, _y, _z):
        k = self.depth % 3
        if k == 0:
            key = 'x'
        elif k == 1:
            key = 'y'
        else:
            key = 'z'

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
        for pos in positions:
            self.addNode(pos[0], pos[1], pos[2])

    def getCoordinates(self):
        return [self.pos['x'],self.pos['y'],self.pos['z']]

def build_kdtree(points, depth=0):
    if not points:
        return None

    k = depth % 3
    axis = 'xyz'[k]

    points.sort(key=lambda point: point[k])
    median_index = len(points) // 2
    median_point = points[median_index]

    node = kdtree(*median_point, depth)

    node.left = build_kdtree(points[:median_index], depth + 1)
    node.right = build_kdtree(points[median_index + 1:], depth + 1)

    return node

ROOT = build_kdtree(SPHERES)

######### PLOT FUNCTION #########
def plotTrees(tree, path):
    plt.figure()
    ax = plt.gca()
    for sphere in SPHERES:
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
    return np.sqrt((pointA[0]-pointB[0])**2 + 
                   (pointA[1]-pointB[1])**2 + 
                   (pointA[2]-pointB[2])**2)
    
def findNearestNode(random_point, tree):
    nearest_node = None
    min_dist = np.Inf

    for node_pos, node in tree.items():
        dist = euclideanDistance(node_pos, random_point)

        if dist < min_dist:
            nearest_node = node
            min_dist = dist
        
    return nearest_node

def findNewPoint(random_point, nearest_node):
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

def distPoint2Line(A,B,C):
    AB = B.T - A.T
    AC = C - A

    if AC.dot(AB) <= 0:
        return np.linalg.norm(AC) > RAD_OBST
    
    BC = C - B

    if BC.dot(AB) >= 0:
        return np.linalg.norm(BC) > RAD_OBST
    
    return (np.linalg.norm(np.cross(AB,AC)) / np.linalg.norm(AB))  > RAD_OBST

def collisionCheck(pointA, pointB):
    min_cor = [min(pointA[i],pointB[i]) for i in range(3)]
    max_cor = [max(pointA[i],pointB[i]) for i in range(3)]

    potentialCollisionSpheres = ROOT.findNodesinRange(min_cor, max_cor)

    for sphere in potentialCollisionSpheres:
        if distPoint2Line(pointA, pointB, sphere) == False:
            return False
    return True

def addNewPoint(new_point, tree):
    neighbours = {}
    min_path_length = np.Inf
    predecessor = None

    for node in tree.values():
        node_pos = node.getNpCoordinates()
        if not collisionCheck(node_pos, new_point):
            continue
        
        edge_length = euclideanDistance(new_point, node_pos)
        path_length = node.min_path_length + edge_length

        if edge_length < RAD_NBR:
            neighbours[node] = edge_length

            if path_length < min_path_length:
                min_path_length = path_length
                predecessor = node

    if not neighbours:
        return None, None

    new_node = Vertex(new_point, predecessor, min_path_length)
    tree[tuple(new_point)] = new_node
    # tree[predecessor.getTupleCoordinates()].successor = new_node

    return neighbours, new_node

def updateNeighbours(neighbours, new_node, tree):
    for nbr_node, edge_length in neighbours.items():
        nbr_position = nbr_node.getTupleCoordinates()
        if (new_node.min_path_length + edge_length) < nbr_node.min_path_length:
            tree[nbr_position].predecessor = new_node
            tree[nbr_position].min_path_length = new_node.min_path_length + edge_length
                    
def returnPartialPath(node, end, depth=0):
    if node == node.predecessor:
        print("Node has itself as predecessor!")

    if node.getTupleCoordinates() == end:
        return [end]
    
    path = returnPartialPath(node.predecessor, end, depth+1)
    path.append(node.getTupleCoordinates())

    return path
    
def findShortestPath(tree):
    min_dist = np.Inf
    nearest_node = None

    for node_pos, node in tree.items():
        dist = euclideanDistance(END_POINT, node_pos)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node

    while nearest_node.getTupleCoordinates() == nearest_node.predecessor.getTupleCoordinates():
        nearest_node = nearest_node.predecessor
    
    return returnPartialPath(nearest_node, START_POINT)

def RRT(N = 700):
    tree = {START_POINT: Vertex(START_POINT)}

    for i in range(N):
        if i%50==0: print("Iteration: ", i)

        if i%25 == 0: x,y,z = END_POINT
        else:
            x = round(random.uniform(X_BOUNDS[0], X_BOUNDS[1]),1)
            y = round(random.uniform(Y_BOUNDS[0], Y_BOUNDS[1]),1)
            z = round(random.uniform(Z_BOUNDS[0], Z_BOUNDS[1]),1)

        random_point = np.array([x,y,z])

        # Find nearest node to random point in current tree (Nearest)
        nearest_node = findNearestNode(random_point, tree)

        # Find new point for current tree (Steer)
        new_point = findNewPoint(random_point, nearest_node)
        if new_point is None:
            continue

        # Add new_point to tree and find its neighbours
        neighbours, new_node = addNewPoint(new_point, tree)
                
        if not neighbours:
            continue

        updateNeighbours(neighbours, new_node, tree)

    return tree

def generate_waypoints(waypoints, num_steps):
    WAYPOINTS = []
    for i in range(len(waypoints) - 1):
        start = np.array(list(waypoints[i]))
        end = np.array(list(waypoints[i + 1]))

        interpolated_points = [start + i * (end - start) / (num_steps + 1) for i in range(1, num_steps + 1)]

        WAYPOINTS.extend(interpolated_points)

    return np.array(WAYPOINTS)

def calcTotalPathLength(path):
    return sum([math.sqrt((path[i][0]-path[i-1][0])**2 + 
                    (path[i][1]-path[i-1][1])**2 + 
                    (path[i][2]-path[i-1][2])**2)
                    for i in range(1,len(path))])

def smoothenPath(path, N):
    x = [path[i][0] for i in range(len(path))]
    y = [path[i][1] for i in range(len(path))]
    z = [path[i][2] for i in range(len(path))]

    tck, u = splprep([x, y, z], s=0, k=2)

    u_fine = np.linspace(0, 1, N)
    x_smooth, y_smooth, z_smooth = splev(u_fine, tck)

    # original_path = np.vstack((x, y, z)).T.astype(np.float32)
    smoothed_path = np.vstack((x_smooth, y_smooth, z_smooth)).T.astype(np.float32)
    return smoothed_path

################# AUTOMATE VARS #################
control_freq_hz=DEFAULT_CONTROL_FREQ_HZ
num_drones=DEFAULT_NUM_DRONES

######### RUN RRT* #########
start_time = time.time()
states = RRT()
end_time = time.time()
print("Runtime: ", end_time - start_time)

# plotTrees(states, [])

path = findShortestPath(states)

plotTrees(states, path)

print(path)
waypoints = np.array([[path[i][0], path[i][1], path[i][2]] for i in range(len(path))])
waypoints = path
num_edges = len(waypoints)-1
INIT_XYZS = np.array([waypoints[0]])
INIT_RPYS = np.array([[0,0,0]])

totPathLen = calcTotalPathLength(waypoints)
# NUM_WP = int(totPathLen // 0.006)
NUM_WP = int(totPathLen // 0.004)
PERIOD = NUM_WP // control_freq_hz
DEFAULT_DURATION_SEC = PERIOD

WAYPOINTS = generate_waypoints(waypoints, NUM_WP//num_edges)
# WAYPOINTS = smoothenPath(path, NUM_WP)
TARGET_POS = WAYPOINTS
wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(TARGET_POS[:,0], TARGET_POS[:,1], TARGET_POS[:,2])
plt.show()
################# AUTOMATE VARS #################

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):

    #### Create the environment ################################
    env = CtrlRRT(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    drone_trajectories = [[] for _ in range(num_drones)]
    START = time.time()

    reached_final_waypoint = False

    DRONE_IDS = env.getDroneIds()
    camera_distance = 1
    camera_yaw = 0
    camera_pitch = 30

    prev_point = WAYPOINTS[0]
    for point in WAYPOINTS[1:]:
        line_id = p.addUserDebugLine(lineFromXYZ=prev_point, lineToXYZ=point, lineColorRGB=[1, 0, 0], physicsClientId=PYB_CLIENT)
        prev_point = point

    for i in range(0, int(duration_sec*env.CTRL_FREQ)-1):
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        drone_pos, _ = p.getBasePositionAndOrientation(DRONE_IDS[0], physicsClientId=PYB_CLIENT)
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, drone_pos, physicsClientId=PYB_CLIENT)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    # target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                    target_pos=TARGET_POS[wp_counters[j]],
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()
    
    # Plot the drone trajectories
    if plot_trajectory:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for traj in drone_trajectories:
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
        start = drone_trajectories[0]
        start = np.array(start)
        ax.scatter(start[0],start[1],start[2], color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Trajectories')
        plt.show()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlRRT and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))