import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CartPoleSwingUpEnv(gym.Env):
	"""
	Description:
		A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

	Modification to orginal CartPoleEnv (gym version 0.18.0) to enable swing-up and swing-up + balance:
		- removed done condition for theta. Only done when x out of bounds (x_threshold).
		- changed reward function
		- wrapped theta
		- changed theta observation: theta --> [sin(theta), cos(theta)]
			- making obervation Box(5)
		- added max_steps
		- added balance: whether the system should also attempt to balance after swinging up
	
	Source:
		This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

	Observation: 
		Type: Box(5)
		Num	Observation                 Min         Max
		0	Cart Position             -4.8            4.8
		1	Cart Velocity             -Inf            Inf
		2	Pole sin(angle)           -1        	  1
		3	Pole cos(angle)      	  -1              1
		4	Pole Velocity At Tip      -4*pi           4*pi
		
	Actions:
		Type: Discrete(2)
		Num	Action
		0	Push cart to the left
		1	Push cart to the right
		
		Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

	Reward:
		...See code. Reward shaping could improve performace.  

	Starting State:
		If the task includes balancing, the pole starts upright 50% of the time. Otherwise the pole starts by hanging down.

	Episode Termination:
		Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
		Episode length is greater than 500 or chosen max_steps.
	
	Solved Requirements
		Considered solved when the system is able to swing-up and balance the pole.
	"""
	
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	MAX_VEL_THETA = 4 * np.pi

	def __init__(self, poleLen=0.5, max_steps=1000, balance=False):
		self.gravity = 9.8
		self.masscart = 1.0
		self.masspole = 0.1
		self.total_mass = (self.masspole + self.masscart)
		self.length = poleLen # 0.5 # actually half the pole's length
		self.polemass_length = (self.masspole * self.length)
		self.force_mag = 10.0
		self.tau = 0.02  # seconds between state updates
		self.kinematics_integrator = 'euler'

		# Angle at which to fail the episode
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4

		# *===== ADDED OBSERVATION =====
		high = np.array([self.x_threshold * 2,
						np.finfo(np.float32).max,
						1.0,
						1.0,
						self.MAX_VEL_THETA],
					dtype=np.float32)
		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)

		self.seed()
		self.viewer = None
		self.state = None
		self.steps_beyond_done = None

		# *===== ADDED =====
		self.balance = balance # whether to also attept to balance or not
		self.n_steps = 0
		self.max_steps = max_steps

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		err_msg = "%r (%s) invalid" % (action, type(action))
		assert self.action_space.contains(action), err_msg

		x, x_dot, theta, theta_dot = self.state
		force = self.force_mag if action == 1 else -self.force_mag
		costheta = math.cos(theta)
		sintheta = math.sin(theta)

		# For the interested reader:
		# https://coneural.org/florian/papers/05_cart_pole.pdf
		temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
		thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
		xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

		if self.kinematics_integrator == 'euler':
			x = x + self.tau * x_dot
			x_dot = x_dot + self.tau * xacc
			theta = theta + self.tau * theta_dot
			theta_dot = theta_dot + self.tau * thetaacc
		else:  # semi-implicit euler
			x_dot = x_dot + self.tau * xacc
			x = x + self.tau * x_dot
			theta_dot = theta_dot + self.tau * thetaacc
			theta = theta + self.tau * theta_dot

		# *===== ADDED WRAP AND BOUND =====
		theta = wrap(theta, -np.pi, np.pi)
		theta_dot = bound(theta_dot, -self.MAX_VEL_THETA, self.MAX_VEL_THETA)
				
		self.state = (x,x_dot,theta,theta_dot)
		
		# *===== REMOVED THETA DONE CONDITION AND CHANGED REWARDING =====
		self.n_steps += 1
		if (self.n_steps > self.max_steps):
			reward = 0
			done = True
		elif (x < -self.x_threshold or x > self.x_threshold):
			reward = -3000.0
			done = True
		elif self.balance:
			reward = 2.0*(1.0 + costheta)
			done = False
		else:
			if costheta > 0.85:
				reward = 2
				done = True
			else:
				reward = -1
				done = False

		if done:
			self.steps_beyond_done = 0
		if self.steps_beyond_done is not None:
			if self.steps_beyond_done == 1:
				logger.warn(
					"You are calling 'step()' even though this "
					"environment has already returned done = True. You "
					"should always call 'reset()' once you receive 'done = "
					"True' -- any further steps are undefined behavior."
				)
			self.steps_beyond_done += 1

		# return np.array(self.state), reward, done, {}
		return self._get_obs(self.state), reward, done, {}

	# *===== CREATED OBSERVATION FUNCTION =====
	def _get_obs(self, s):
		return np.array([s[0], s[1], np.cos(s[2]), np.sin(s[2]), s[3]])

	def reset(self):
		self.n_steps = 0
		self.steps_beyond_done = None
		self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		# *=== ADDED INIT PENDULUM HANGING DOWN ===
		if self.balance:
			if np.random.rand() > 0.5:
				self.state[2] = np.pi
		else:
			self.state[2] = np.pi
		# return np.array(self.state)
		return self._get_obs(self.state)

	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.x_threshold * 2
		scale = screen_width/world_width
		carty = 100  # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
			axleoffset = cartheight / 4.0
			cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
			pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
			pole.set_color(.8, .6, .4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5, .5, .8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0, carty), (screen_width, carty))
			self.track.set_color(0, 0, 0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		if self.state is None:
			return None

		# Edit the pole polygon vertex
		pole = self._pole_geom
		l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
		pole.v = [(l, b), (l, t), (r, t), (r, b)]

		x = self.state
		cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None


# *===== ADDED WRAP AND BOUND FUNCTIONS FROM ACROBOT =====
def wrap(x, m, M):
	"""Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
	truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
	For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

	Args:
		x: a scalar
		m: minimum possible value in range
		M: maximum possible value in range

	Returns:
		x: a scalar, wrapped
	"""
	diff = M - m
	while x > M:
		x = x - diff
	while x < m:
		x = x + diff
	return x

def bound(x, m, M=None):
	"""Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
	have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

	Args:
		x: scalar

	Returns:
		x: scalar, bound between min (m) and Max (M)
	"""
	if M is None:
		M = m[1]
		m = m[0]
	# bound x between min (m) and Max (M)
	return min(max(x, m), M)