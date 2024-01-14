import numpy as np

MAX_DISTANCE = 0.1
TIMESTEP = 1/48

class Drone:
	def __init__(self, planner, obstacle_avoidance = None):
		self.planner = planner
		self.obstacle_avoidance = obstacle_avoidance

		states = self.planner.solve()
		self.path = iter(self.planner.path)
		self.p = next(self.path)
		self.goal = next(self.path)

		self.destionation_reached = False
		
	def generate_waypoint(self, obstacles):
		"""
		Find next waypoint based on the wall closeby and the other drones
		"""
		if np.linalg.norm(self.p - self.goal) < MAX_DISTANCE:
			try:
				self.goal = next(self.path)
			except:
				# We reached our goal
				pass

			self.obstacle_avoidance.step(obstacles, max_samples = 100)
			return self.obstacle_avoidance.speed_to_waypoint(t=TIMESTEP)
		else:
			return self.goal

	def generate_velocity_waypoint(self, obstacles, D2=False):
		"""
		Find next waypoint based on the wall closeby and the other drones
		"""
		if np.linalg.norm(self.p - self.goal) < MAX_DISTANCE:
			try:
				self.goal = next(self.path)
			except:
				if not self.destionation_reached:
					self.obstacle_avoidance.vmax /= 10
				self.destionation_reached = True
		else:
			self.obstacle_avoidance.step(obstacles, max_samples = 100, D2=D2)
		return self.v
		#return self.obstacle_avoidance.desired_vel

	@property
	def p(self):
		return self.obstacle_avoidance.p

	@p.setter
	def p(self, value):
		self.obstacle_avoidance.p = np.array(value)

	@property
	def v(self):
		return self.obstacle_avoidance.v

	@v.setter
	def v(self, value):
		self.obstacle_avoidance.v = np.array(value)

	@property
	def goal(self):
		return self.obstacle_avoidance.goal

	@goal.setter
	def goal(self, value):
		self.obstacle_avoidance.goal = np.array(value)