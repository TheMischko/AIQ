from gym.envs.registration import register
from importlib_metadata import entry_points

__all__ = ["BF","BFGym"]  # add others to this list


register(
    id = 'BF-v0',
    entry_point = 'refmachines:BF',
    nondeterministic = False
)