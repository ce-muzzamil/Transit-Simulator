import gymnasium as gym

class TransitNetworkEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def reset(self, *, seed = None, options = None):
        return super().reset(seed=seed, options=options)

    def step(self, action):
        return super().step(action)

    def observation(self):
        pass
    
    def reward(self):
        pass

    def render(self):
        pass