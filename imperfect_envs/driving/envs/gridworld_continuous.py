import io
from typing import Text
import gym
from gym import spaces
from PIL import Image
import numpy as np
import scipy.special
from driving.world import World
from driving.entities import TextEntity, Entity
from driving.agents import Car, Building, Goal
from driving.geometry import Point
from typing import Tuple

import random

class PidVelPolicy:
    """PID controller for H that maintains its initial velocity."""

    def __init__(self, dt: float, params: Tuple[float, float, float] = (3.0, 1.0, 6.0)):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []
        self.dt = dt
        self.Kp, self.Ki, self.Kd = params

    def action(self, obs):
        my_y_dot = obs[3]
        if self._target_vel is None:
            self._target_vel = my_y_dot
        error = self._target_vel - my_y_dot
        derivative = (error - self.previous_error) * self.dt
        self.integral = self.integral + self.dt * error
        acc = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        self.errors.append(error)
        return acc

    def reset(self):
        self._target_vel = None
        self.previous_error = 0
        self.integral = 0
        self.errors = []

    def __str__(self):
        return "PidVelPolicy({})".format(self.dt)

class GridworldContinuousEnv(gym.Env):

    def __init__(self,
                 dt: float = 0.1,
                 width: int = 30,
                 height: int = 40,
                 time_limit: float = 300.0):
        super(GridworldContinuousEnv, self).__init__()
        self.dt = dt
        self.width = width
        self.height = height
        self.world = World(self.dt, width=width, height=height, ppm=6)
        self.accelerate = PidVelPolicy(self.dt)
        self.step_num = 0
        self.time_limit = time_limit
        self.action_space = spaces.Box(
            np.array([-1.]), np.array([1.]), dtype=np.float32
        )
        self.goal_radius = 2.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,))
        self.start = np.array([self.width/2.,self.goal_radius])
        self.goal = np.array([self.width/2., self.height-self.goal_radius])
        self.max_dist = np.linalg.norm(self.goal-self.start,2)

        self.target = [self.height/5., self.height*2./5., self.height*3./5., self.height*4./5., np.inf]
        self.obstacle_width = 6.
        self.initial_speed = 3.

    def step(self, action: np.ndarray, verbose: bool = False):
        self.step_num += 1

        action = action * 0.1
        car = self.world.dynamic_agents[0]
        acc = self.accelerate.action(self._get_obs())
        action = np.append(action, acc)
        if self.stop:
            action = np.array([0, -5])
        car.set_control(*action)
        self.world.tick()

        reward = self.reward(verbose)

        done = False
        if car.y >= self.height or car.y <= 0 or car.x <= 0 or car.x >= self.width:
            reward -= 10000
            done = True
        if self.step_num >= self.time_limit:
            done = True
        if self.car.collidesWith(self.goal_obj):
            done = True
            self.stop = True
        #if self.step_num < 6:
        #    done = False
        return self._get_obs(), reward, done, {'episode': {'r': reward, 'l': self.step_num}}

    def reset(self):
        self.world.reset()
        self.stop = False
        self.target_count = 0

        self.buildings = [
            Building(Point(self.width/2., self.height/2.-3), Point(self.obstacle_width,1), "gray80"),
        ]

        random_dis = random.random()*2.
        random_angle = random.random()*2*np.pi
        init_x = self.start[0] + random_dis*np.cos(random_angle)
        init_y = self.start[1] + random_dis*np.sin(random_angle)
        self.car = Car(Point(init_x, init_y), np.pi/2., "blue")
        self.car.velocity = Point(0, self.initial_speed)

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), self.goal_radius, 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

    def reset_with_obs(self, obs):
        self.world.reset()
        self.stop = False
        self.target_count = 0

        self.buildings = [
            Building(Point(self.width/2., self.height/2.-3), Point(self.obstacle_width,1), "gray80"),
        ]

        init_x = (obs[0]/2.+0.5)*self.width
        init_y = (obs[1]/2.+0.5)*self.height
        self.car = Car(Point(init_x, init_y), np.pi/2., "blue")
        self.car.velocity = Point(0, self.initial_speed)

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), self.goal_radius, 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Get state of car
        """
        return_state = np.array(self.world.state)
        #print(return_state)
        return_state[1] = 2.* ((return_state[1] / self.height) - 0.5)
        return_state[0] = 2.* ((return_state[0] / self.width) - 0.5)
        return_state[2] /= self.initial_speed
        return_state[3] /= self.initial_speed
        return return_state

    def inverse_dynamic(self, state, next_state):
        return (next_state[-2] / np.linalg.norm(self.initial_speed*state[2:4], ord=2))/self.dt

    def reward(self, verbose, weight=10.0):
        dist_rew = -1. # * (self.car.center.distanceTo(self.goal_obj)/self.max_dist)
        coll_rew = 0
        for building in self.buildings:
            if self.car.collidesWith(building):
                coll_rew = -1000.
                break

        goal_rew = 0.0
        if self.car.collidesWith(self.goal_obj) and (not self.stop):
            goal_rew = 100.

        extra_rew = 0.
        #if self.car.x < self.width / 4.:
        #    extra_rew = (self.width / 4. - self.car.x)/(self.width/4.) * (-1.)
        #elif self.car.x > self.width * 3. / 4.:
        #    extra_rew = (self.car.x-self.width * 3. / 4.)/(self.width/4.) * (-1.)

        reward = sum([dist_rew, coll_rew, extra_rew, goal_rew])
        if verbose: print("dist reward: ", dist_rew,
                          "goal reward: ", goal_rew,
                          "extra reward: ", extra_rew,
                          "reward: ", reward)
        return reward

    def render(self):
        self.world.render()

class GridworldContinuousSlowRandomInitEnv(GridworldContinuousEnv):
    def reset(self):
        self.world.reset()

        self.stop = False
        self.target_count = 0

        self.buildings = [
            Building(Point(self.width/2., self.height/2.-3), Point(self.obstacle_width,1), "gray80"),
        ]

        while True:
            random_w = random.random()
            random_h = random.random()
            init_x = self.width/2.-(self.obstacle_width/2.+2.) + random_w*(self.obstacle_width+4.)
            init_y = self.goal_radius + (self.height-3*self.goal_radius)*random_h
            cond1 = abs(init_x - self.width/2.) < (self.obstacle_width/2.+2.) and init_y-self.height/2. < 3. and init_y-self.height/2.>-13.
            slope = ((self.height - self.goal_radius) - (self.height/2.-3))/(self.width/4.)
            #print(slope, init_x, ((self.width/4.-abs(init_x - self.width/2.)) * slope + (self.height/2.-3.)))
            cond2 = init_y < ((self.width/4.-abs(init_x - self.width/2.)) * slope + (self.height/2.-3.))
            if cond2 and not cond1:
                break
        init_heading = np.pi/2. # np.arctan2(self.goal[1] - init_y, self.goal[0]-init_x)
        self.car = Car(Point(init_x, init_y), init_heading, "blue")
        self.car.velocity = Point(0, self.initial_speed)

        self.goal_obj = Goal(Point(self.goal[0], self.goal[1]), self.goal_radius, 0.0)

        for building in self.buildings:
            self.world.add(building)
        self.world.add(self.car)
        self.world.add(self.goal_obj)
        
        self.last_heading = np.pi / 2

        self.step_num = 0
        return self._get_obs()

class GridworldContinuousFastRandomInitEnv(GridworldContinuousSlowRandomInitEnv):
    def __init__(self,
                 dt: float = 0.1,
                 width: int = 30,
                 height: int = 40,
                 time_limit: float = 300.0):
        super(GridworldContinuousFastRandomInitEnv, self).__init__(dt, width, height, time_limit)
        self.initial_speed = 9.
