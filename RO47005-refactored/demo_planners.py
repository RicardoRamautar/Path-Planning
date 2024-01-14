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
from environment.variables import RAD_QUAD, RAD_SPHERE, RAD_NP, RAD_NBR, ENVIRONMENT, ENVIRONMENT_SPHERES
from planners.RRT_star import RRT_star
from planners.Dijkstra import Dijkstra
from planners.Bidirectional_RRT import Bidirectional_RRT

from drones.Drone import Drone
from obstacle_avoidance.velocity_obstacles_3D import * 
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

import pickle

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
DEFAULT_PLANNER = "RRT_star"
plot_trajectory = True
planners = {"Dijkstra": Dijkstra, "RRT_star": RRT_star, "Bidirectional_RRT": Bidirectional_RRT}

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

DEFAULT_DURATION_SEC = 100

INIT_XYZS = np.array([START_POINT])
INIT_RPYS = np.array([[0,0,0]])

################# AUTOMATE VARS #################
control_freq_hz=DEFAULT_CONTROL_FREQ_HZ

######### RUN RRT* #########



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
        colab=DEFAULT_COLAB,
        planner=DEFAULT_PLANNER,
        backup=False
        ):
    
    if backup:
        with open('drones.pkl', 'rb') as file:
            drones = pickle.load(file)

    else:
        planner = planners[planner](START_POINT, END_POINT, [X_BOUNDS, Y_BOUNDS, Z_BOUNDS], SPHERES)
        avoidance = Agent2d(START_POINT, RAD_QUAD, max_speed=0.26, time_to_collision=1)
        drones = [Drone(planner, avoidance)]

        with open('drones.pkl', 'wb') as file:
            pickle.dump(drones, file)

    num_drones= len(drones)

    #### Create the environment ################################
    env = VelocityAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=False,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    drone_trajectories = [[] for _ in range(num_drones)]
    START = time.time()

    reached_final_waypoint = False

    DRONE_IDS = env.getDroneIds()
    camera_distance = 0.2
    camera_yaw = 0
    camera_pitch = 30

    if plot_trajectory:
        waypoints = drones[0].planner.path
        prev_point = waypoints[0]
        for point in waypoints[1:]:
            line_id = p.addUserDebugLine(lineFromXYZ=prev_point, lineToXYZ=point, lineColorRGB=[1, 0, 0], physicsClientId=PYB_CLIENT)
            prev_point = point

    if obstacles:
        p.loadURDF(str(ENVIRONMENT_SPHERES))


    for i in range(0, int(duration_sec*env.CTRL_FREQ)-1):
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        drone.p, _ = p.getBasePositionAndOrientation(DRONE_IDS[0], physicsClientId=PYB_CLIENT)
        cam_info = p.getDebugVisualizerCamera()
        p.resetDebugVisualizerCamera(cam_info[10], cam_info[8], cam_info[9], drone.p, PYB_CLIENT)

        #### Compute control for the current way point #############
        for j, drone in enumerate(drones):
            range_obstacles = 0.1
            obstacles = [Vehicle2d(pos, RAD_SPHERE) for pos in drone.planner.ROOT.findNodesinRange(drone.p - range_obstacles * np.ones(3), drone.p + range_obstacles * np.ones(3))]
            obstacles += [drone.obstacle_avoidance for drone in  drones[:j] + drones[j+1:]]
            velocity = drone.generate_velocity_waypoint(obstacles)
            speed = np.linalg.norm(velocity)
            drone_velocity = np.hstack([velocity / speed, drone.obstacle_avoidance.vmax / speed])
            action[j, :] = drone_velocity
            
        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlRRT and DSLPIDControl')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to render the obstacles (default: True)', metavar='')
    parser.add_argument('--planner',          default=DEFAULT_PLANNER,       type=str,      help='Planner between RRT_star, Bidirectional_RRT and Dijkstra (default: Dijkstra)', metavar='')
    parser.add_argument('--backup',          default=False,       type=str2bool,      help='Whether to load the last path found (default: False)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))