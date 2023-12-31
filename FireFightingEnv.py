from typing import Tuple

# from gym.core import ActType, ObsType
import numpy as np
import AgentState
import MapEnv
# import gymnasium as gym
# from gymnasium import spaces


class FFEnv():
    def __init__(self, agent_num):
        self.mapEnv = MapEnv.MapEnv(BIRTH=(2, 3, agent_num), SIZE=(15,15))
        self.agent_num = agent_num
        self.agents = []
        self.rewards = np.zeros(agent_num)
        self.observe_space = []
        self.finished = 0
        self.curr_step = 0
        self.max_step = 200
        self.reset()

    def reset(self, **kwargs):
        self.mapEnv.reset()
        self.agents.clear()
        self.rewards = np.zeros(self.agent_num)
        self.observe_space.clear()
        self.finished = 0
        self.curr_step = 0
        init_pose = self.mapEnv.getInitPose()
        for i in range(self.agent_num):
            self.agents.append(AgentState.AgentState(self.mapEnv, init_pose[i][0], init_pose[i][1]))
            self.observe_space.append(self.agents[i].observe())
        return self.observe_space

    def step(self, action):
        """
        One step of the environment and all agents
        :param action: A list of actions for each agent
        :return: The observation of all agents, reward of all agents, if the sequence is finished
        """
        for i, a in enumerate(self.agents):
            self.observe_space[i], self.rewards[i] = a.step(action[i])

        self.mapEnv.mapUpdate()
        self.curr_step += 1
        if np.sum(self.mapEnv.getFire()) == 0:
            for i, a in enumerate(self.rewards):
                self.rewards[i] = AgentState.Reward.REWARD_EPOCH_SUCCESS
            self.finished = 1
        elif np.sum(self.mapEnv.getHP()) < np.sum(self.mapEnv.getHPInit()) / 3:
            for i, a in enumerate(self.rewards):
                self.rewards[i] = AgentState.Reward.REWARD_EPOCH_UNSUCCESSFUL
            self.finished = 1
        elif self.curr_step > self.max_step:
            for i, a in enumerate(self.rewards):
                self.rewards[i] = AgentState.Reward.REWARD_EPOCH_UNSUCCESSFUL
            self.finished = 1
        else:
            self.finished = 0 # TODO: Here is a bug!
        # print(np.sum(self.mapEnv.getHP()))
        # print(np.sum(self.mapEnv.getHPInit()) / 3)
        # print(self.mapEnv.getHP())
        return self.observe_space, self.rewards, self.finished

if __name__ == "__main__":
    env = FFEnv(agent_num=3)

    actions = []
    for i in range(3):
        actions.append(dict())

    for i in range(5):
        for a in actions:
            a[0] = np.random.choice([0, 1, 2, 3, 4])
            a[1] = np.random.choice([0, 1])

        env.step(actions)
        
        env.mapEnv.plotAgent()
        input("Press any key")
