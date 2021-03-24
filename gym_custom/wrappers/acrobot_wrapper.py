class AcrobotWrapper(gym.Wrapper):
    '''Acrobot
        - inputs: Discrete(3) [0,1,2]
        - observations: Box(6)
    '''

    tasks = ["swing-up",
             "balance",
             "swing-up balance"]

    def __init__(self, env, task, max_steps=500): #, max_episodes=400):
        super().__init__(env)
        self.task = task
        self.n_steps = 0
        self.max_steps = max_steps

    def reset(self):
        self.n_steps = 0
        obs = self.env.reset()
        if self.task == "balance":
            self.env.state[0] = np.pi
            self.env.state[1] = np.random.uniform(low=-0.005,high=0.005)
            self.env.state[2] = 0
            self.env.state[3] = 0
            s = self.env.state
            return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
        return obs

    def step(self, action):
        self.n_steps += 1
        obs, reward, done, info = self.env.step(action)
        done = self._terminal()
        reward = self._get_reward(done)
        return (obs, reward, done, {})

    def _get_reward(self, done):
        if self.task == "balance" and not done:
            return 1
        else:
            if done:
                return 2
            s = self.env.state
            if bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.8):
                return 1
            elif bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.5):
                return 0
            else:
                return -1

    def _terminal(self):
        if self.task == "balance":
            s = self.env.state
            return bool(-cos(s[0]) - cos(s[1] + s[0]) < 1.8)        # done when falls below
        else:
            if self.n_steps > self.max_steps:
                return True
            elif self.task == "swing-up":
                s = self.env.state
                return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.9)    # done when above
            else:
                return False                                        # only done when max_steps