import dmlab2d
from gymnasium import spaces
import numpy as np
from ray.rllib.env import multi_agent_env
from meltingpot_wrapper import MeltingPotEnv

from baselines.train import utils

class CustomRewards(MeltingPotEnv): 
  def step(self, action_dict):
    """See base class."""
    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    rewards = {
        agent_id: timestep.reward[index]
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }
    for id in rewards:
      print(rewards[id], end=" || ")
    print()
    done = {'__all__': timestep.last()}
    info = {}

    observations = utils.timestep_to_observations(timestep)
    return observations, rewards, done, done, info