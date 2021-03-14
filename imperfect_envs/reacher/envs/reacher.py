import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gym
import os
import random

class ReacherCustomEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, config_file='reacher.xml'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, ('%s/assets/'+config_file) % dir_path, 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        reward_for_eval = reward_dist * 10# - np.sqrt(self.sim.data.qvel.flat[0]**2+self.sim.data.qvel.flat[1]**2) / 20.

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, reward_eval=reward_for_eval)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_with_obs(self, obs):
        self.sim.reset()
        qpos = np.array([0., 0., 0., 0.])
        self.goal = obs[4:6]
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        qvel[0:2] = obs[6:8]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_model(self):
        #self.close_goal = False
        #qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        #while True:
        #    self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        #    if np.linalg.norm(self.goal) < 0.2:
        #        break
        qpos = np.array([0., 0., 0., 0.])
        self.goal = np.concatenate([self.np_random.uniform(low=-.1, high=.1, size=1),
                                    self.np_random.uniform(low=-.2, high=-.1, size=1) if self.np_random.uniform(low=0, high=1., size=1)[0]>0.5 else self.np_random.uniform(low=.1, high=.2, size=1)])
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

class ReacherCustomAction1Env(ReacherCustomEnv):
    def __init__(self):
        super(ReacherCustomAction1Env, self).__init__('reacher_action1.xml')

class ReacherCustomRAction1Env(ReacherCustomEnv):
    def __init__(self):
        super(ReacherCustomRAction1Env, self).__init__('reacher_action1.xml')
        self.action_space = gym.spaces.Box(low=np.array([-1.,-1.]).astype('float32'), high=np.array([0.,0.]).astype('float32'))

    def step(self, a):
        a = np.clip(a, -1., 0.)
        return super(ReacherCustomRAction1Env, self).step(a)

class ReacherCustomAction2Env(ReacherCustomEnv):
    def __init__(self):
        super(ReacherCustomAction2Env, self).__init__('reacher_action2.xml')

class ReacherCustomRAction2Env(ReacherCustomEnv):
    def __init__(self):
        super(ReacherCustomRAction2Env, self).__init__('reacher_action2.xml')
        self.action_space = gym.spaces.Box(low=np.array([0.,0.]).astype('float32'), high=np.array([1.,1.]).astype('float32'))

    def step(self, a):
        a = np.clip(a, 0., 1.)
        return super(ReacherCustomRAction2Env, self).step(a)

