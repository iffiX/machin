from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import os

_root_dir = os.path.dirname(scenarios.__file__)
_all_files = [
    f.split(".")[0]
    for f in os.listdir(_root_dir)
    if (not f.startswith("__") and f.endswith(".py"))
]


def all_envs():
    return _all_files


def create_env(env_name):
    if env_name not in all_envs():
        raise RuntimeError("Invalid multi-agent environment: " + env_name)
    # load scenario from script
    scenario = scenarios.load(env_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        shared_viewer=False,
    )
    return env
