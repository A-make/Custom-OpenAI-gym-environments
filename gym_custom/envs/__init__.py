from gym.envs.registration import register

# Classic
# ----------------------------------------

register(
    id='CartPoleSwingUp-v0',
    entry_point='gym_custom.envs.classic_control:CartPoleSwingUpEnv',
    max_episode_steps=500,
    reward_threshold=497.0,
)
register(
    id='CartPoleSwingUp2-v0',
    entry_point='gym_custom.envs.classic_control:CartPoleSwingUp2Env',
    max_episode_steps=500,
    reward_threshold=497.0,
)

# Box2d
# ----------------------------------------

register(
    id='RocketLander-v0',
    entry_point='gym_custom.envs.box2d:RocketLander',
    max_episode_steps=1000,
    reward_threshold=0,
)