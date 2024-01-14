"""Script demonstrating the joint use of velocity input.

The simulation is run by a `VelocityAviary` environment.

Example
-------
In a terminal, run as:

    $ python pid_velocity.py

Notes
-----
The drones use interal PID control to track a target velocity.

"""
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

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

from drones.Drone import Drone
from obstacle_avoidance.velocity_obstacles_3D import * 
from environment.variables import RAD_QUAD, RAD_OBST, RAD_NP, RAD_NBR, ENVIRONMENT, ENVIRONMENT_SPHERES

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
TAU = 1/DEFAULT_CONTROL_FREQ_HZ
DEFAULT_DURATION_SEC = 80
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

class Fake_planner:
    def __init__(self):
        self.path = (np.array([0, 0, 0.3]), np.array([1.3, 1.3, 0.3]))

    def solve(self, *args):
        pass

def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):

    lines = []
        #### Initialize the simulation #############################
    INIT_XYZS = np.array([
                          [ 0, 0, .3],
                          [.35, 0, .3],
                          [.8, 0, .3],
                          [1, 0, .3]
                          ])
    INIT_RPYS = np.array([
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]
                          ])
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VelocityAviary(drone_model=drone,
                         num_drones=4,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
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
    DRONE_IDS = env.getDroneIds()
    drones = [Drone(Fake_planner(), Agent2d(INIT_XYZS[i, :], RAD_QUAD, max_speed=0.26, time_to_collision=50)) for i in range(4)]

    p.addUserDebugLine(lineFromXYZ=drones[0].planner.path[0], lineToXYZ=drones[0].planner.path[1], lineColorRGB=[1, 0, 0], physicsClientId=PYB_CLIENT)
    cam_info = p.getDebugVisualizerCamera()
    #### Compute number of control steps in the simlation ######
    PERIOD = duration_sec
    NUM_WP = control_freq_hz*PERIOD
    wp_counters = np.array([0 for i in range(4)])

    #### Initialize the velocity target ########################
    TARGET_VEL = np.zeros((4,NUM_WP,4))
    for i in range(NUM_WP):
        TARGET_VEL[1, i, :] = [0, 1, 0, .6]# if i < (NUM_WP/8+NUM_WP/6) else [0, -1, 0, 0.99]
        TARGET_VEL[2, i, :] = [0, 1, .39, 1]# if i < (NUM_WP/8+2*NUM_WP/6) else [-0.2, -1, 0.2, 0.99]
        TARGET_VEL[3, i, :] = [0, 1, .15, .8]# if i < (NUM_WP/8+3*NUM_WP/6) else [0, -1, 0.5, 0.99]

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=4,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Run the simulation ####################################
    action = np.zeros((4,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        ############################################################
        # for j in range(3): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        for j in range(4):
            drones[j].p, _ = p.getBasePositionAndOrientation(DRONE_IDS[j], physicsClientId=PYB_CLIENT)
            drones[j].v, _ = p.getBaseVelocity(DRONE_IDS[j], physicsClientId=PYB_CLIENT)
        cam_info = p.getDebugVisualizerCamera()
        p.resetDebugVisualizerCamera(0.35, cam_info[9], cam_info[8], drones[0].p, PYB_CLIENT)

        #### Compute control for the current way point #############

        obstacles = [drone.obstacle_avoidance for drone in drones[1:]]
        velocity = drones[0].generate_velocity_waypoint(obstacles, D2=False)
        speed = np.linalg.norm(velocity)
        TARGET_VEL[0, wp_counters[j], :] = np.hstack([velocity / speed, drones[0].obstacle_avoidance.vmax / speed])
        
        for j in range(4):
            action[j, :] = TARGET_VEL[j, wp_counters[j], :]

        #### Go to the next way point and loop #####################
        for j in range(4):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(4):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state= obs[j],
                       control=np.hstack([TARGET_VEL[j, wp_counters[j], 0:3], np.zeros(9)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Velocity control example using VelocityAviary')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: cf2p)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,      type=str2bool,      help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))