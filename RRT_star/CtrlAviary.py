import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

import json


class CtrlAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )
        if obstacles:
            self._addObstacles()


    ################################################################################

    
    def _addObstacles(self):
        # num_spheres = 5  # Number of spheres to create
        sphere_radius = 0.5  # Radius of the spheres
        sphere_color = [1, 1, 1, 1]  # Red color in RGBA format

        # sphere_positions = np.array([[0,0,1],[0,1,1],[0,2,1],[0,3,1],[0,4,1],[0,5,1],[0,6,1],[0,7,1],[0,8,1],[0,9,1],
        #         [9,0,1],[9,1,1],[9,2,1],[9,3,1],[9,4,1],[9,5,1],[9,6,1],[9,7,1],[9,8,1],[9,9,1],
        #         [1,1,1],[2,1,1],[7,1,1],[8,1,1],[9,1,1],[3,3,1],[4,3,1],[5,3,1],[6,3,1],[7,3,1],
        #         [8,3,1],[1,6,1],[2,6,1],[3,6,1],[4,6,1],[1,9,1],[2,9,1],[3,9,1],[6,9,1],[7,9,1],
        #         [8,9,1],[0,0,2],[0,1,2],[0,2,2],[0,3,2],[0,4,2],[0,5,2],[0,6,2],[0,7,2],[0,8,2],[0,9,2],
        #         [9,0,2],[9,1,2],[9,2,2],[9,3,2],[9,4,2],[9,5,2],[9,6,2],[9,7,2],[9,8,2],[9,9,2],
        #         [1,1,2],[2,1,2],[7,1,2],[8,1,2],[9,1,2],[3,3,2],[4,3,2],[5,3,2],[6,3,2],[7,3,2],
        #         [8,3,2],[1,6,2],[2,6,2],[3,6,2],[4,6,2],[1,9,2],[2,9,2],[3,9,2],[6,9,2],[7,9,2],
        #         [8,9,2],[0,0,3],[0,1,3],[0,2,3],[0,3,3],[0,4,3],[0,5,3],[0,6,3],[0,7,3],[0,8,3],[0,9,3],
        #         [9,0,3],[9,1,3],[9,2,3],[9,3,3],[9,4,3],[9,5,3],[9,6,3],[9,7,3],[9,8,3],[9,9,3],
        #         [1,1,3],[2,1,3],[7,1,3],[8,1,3],[9,1,3],[3,3,3],[4,3,3],[5,3,3],[6,3,3],[7,3,3],
        #         [8,3,3],[1,6,3],[2,6,3],[3,6,3],[4,6,3],[1,9,3],[2,9,3],[3,9,3],[6,9,3],[7,9,3],
        #         [8,9,3],[0,0,0],[0,1,0],[0,2,0],[0,3,0],[0,4,0],[0,5,0],[0,6,0],[0,7,0],[0,8,0],[0,9,0],
        #         [1,0,0],[1,1,0],[1,2,0],[1,3,0],[1,4,0],[1,5,0],[1,6,0],[1,7,0],[1,8,0],[1,9,0],
        #         [2,0,0],[2,1,0],[2,2,0],[2,3,0],[2,4,0],[2,5,0],[2,6,0],[2,7,0],[2,8,0],[2,9,0],
        #         [3,0,0],[3,1,0],[3,2,0],[3,3,0],[3,4,0],[3,5,0],[3,6,0],[3,7,0],[3,8,0],[3,9,0],
        #         [4,0,0],[4,1,0],[4,2,0],[4,3,0],[4,4,0],[4,5,0],[4,6,0],[4,7,0],[4,8,0],[4,9,0],
        #         [5,0,0],[5,1,0],[5,2,0],[5,3,0],[5,4,0],[5,5,0],[5,6,0],[5,7,0],[5,8,0],[5,9,0],
        #         [6,0,0],[6,1,0],[6,2,0],[6,3,0],[6,4,0],[6,5,0],[6,6,0],[6,7,0],[6,8,0],[6,9,0],
        #         [7,0,0],[7,1,0],[7,2,0],[7,3,0],[7,4,0],[7,5,0],[7,6,0],[7,7,0],[7,8,0],[7,9,0],
        #         [8,0,0],[8,1,0],[8,2,0],[8,3,0],[8,4,0],[8,5,0],[8,6,0],[8,7,0],[8,8,0],[8,9,0],
        #         [9,0,0],[9,1,0],[9,2,0],[9,3,0],[9,4,0],[9,5,0],[9,6,0],[9,7,0],[9,8,0],[9,9,0]])
        
        f = open('examples/Implementation/maze.json')
        data = json.load(f)
        sphere_positions = np.array(data['maze'])
        num_spheres = sphere_positions.shape[0]

        # custom_urdf_path="wall1.urdf"
        # p.loadURDF(custom_urdf_path, basePosition=[0, 0, 0])

        
        for i in range(num_spheres):
            sphere_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=sphere_color)
            sphere_position = sphere_positions[i]
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_visual_shape_id, basePosition=sphere_position)

        # sphere_radius = 16.67
        # for pos in sphere_positions:
        #     p.loadURDF("sphere_small.urdf", basePosition=pos, globalScaling=sphere_radius)
            # sphereId = p.loadURDF("sphere_small.urdf", basePosition=pos, globalScaling=sphere_radius,rgbaColor=sphere_color)



    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
