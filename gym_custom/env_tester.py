import time
import numpy as np
import gym
from envs.classic_control.cartpole_swingup import CartPoleSwingUpEnv
from envs.classic_control.double_pendulum import DoublePendulumEnv

# env_original = gym.make('Acrobot-v1')
# env = AcrobotWrapper(env_original)

PRINTING = False

# env = gym.make('Pendulum-v0')
# env = CartPoleSwingUpEnv()
env = DoublePendulumEnv()

obs = env.reset()
action = 0
for i in range(200):
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  if PRINTING:
    print(f'Action:      {action}')
    print(f'Observation: {obs}')
    print(f'Reward:      {reward}')
  env.render()
  time.sleep(0.02)
  if done:
    print(f'Done on index {i}')
    obs = env.reset()
    break
env.close()