from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv

class LastActionObsWrapper(gym.Wrapper):
    """
    Overview:
       Add last frame action to the observation.
    Interface:
        ``__init__``, ``reset``, ``step``, ``_get_obs_with_action``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``observation_space``
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._obs_space = env.observation_space
        assert len(self._obs_space.shape)==3, self._obs_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.min(self._obs_space.low),
            high=np.max(self._obs_space.high),
            shape=(self._obs_space.shape[0]+1, self._obs_space.shape[1], self._obs_space.shape[2]),
            dtype=self._obs_space.dtype
        )
    
    def reset(self):
        reset_action = 0
        obs = self.env.reset()
        return self._get_obs_with_action(reset_action, obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_obs_with_action(action, obs)
        return obs, reward, done, info

    def _get_obs_with_action(self, action, obs):
        scaled_action = action / self.env.action_space.n
        action_obs = np.ones((1, self._obs_space.shape[1], self._obs_space.shape[2]), dtype=np.float32) * scaled_action
        return np.concatenate([obs, action_obs], axis=0)


class StickyActionWrapper(gym.ActionWrapper):
    """
    Overview:
       A certain possibility to select the last action
    Interface:
        ``__init__``, ``action``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``p_sticky``: possibility to select the last action
    """
    def __init__(self, env: gym.Env, p_sticky: float):
        super().__init__(env)
        self.p_sticky = p_sticky
        self.last_action = 0
    
    def action(self, action):
        if np.random.random() < self.p_sticky:
            action = self.last_action
        self.last_action = action
        return action


class SparseRewardWrapper(gym.Wrapper):
    """
    Overview:
       Only death and pass sparse reward
    Interface:
        ``__init__``, ``step``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        dead = True if reward == -15 else False
        reward = 0
        if info['flag_get']:
            reward = 15
        if dead:
            reward = -15
        return obs, reward, done, info


class CoinRewardWrapper(gym.Wrapper):
    """
    Overview:
        add coin reward
    Interface:
        ``__init__``, ``step``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += info['coins'] * 10
        return obs, reward, done, info