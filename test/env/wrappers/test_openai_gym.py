"""
Currently, openai gym offers a great range of environments and we try to
test all kinds of them (not all), however, "robotics" and "mujoco"
requires a license and we cannot test them, but theoretically they
should work just fine.

Submit us a issue if you have found any problem.
"""
from test.util_platforms import linux_only_forall

linux_only_forall()

from machin.env.wrappers import openai_gym
from random import choice, sample
from colorlog import getLogger
import pytest
import gym
import numpy as np

logger = getLogger("default")
ENV_NUM = 2
SAMPLE_NUM = 2
WORKER_NUM = 2


def mock_action(action_space: gym.spaces.Space):
    return action_space.sample()


def prepare_envs(env_list):
    for env in env_list:
        env.reset()


def should_skip(spec):
    # From gym/envs/tests/spec_list.py
    # Used to check whether a gym environment should be tested.

    # We skip tests for envs that require dependencies or are otherwise
    # troublesome to run frequently
    ep = spec.entry_point

    # No need to test unittest environments
    if ep.startswith("gym.envs.unittest"):
        return True

    # Skip not renderable tests
    if ep.startswith("gym.envs.algorithmic") or ep.startswith("gym.envs.toy_text"):
        return True

    # Skip mujoco tests
    if ep.startswith("gym.envs.mujoco") or ep.startswith("gym.envs.robotics:"):
        return True
    try:
        import atari_py
    except ImportError:
        if ep.startswith("gym.envs.atari"):
            return True
    try:
        import Box2D
    except ImportError:
        if ep.startswith("gym.envs.box2d"):
            return True

    if (
        "GoEnv" in ep
        or "HexEnv" in ep
        or (
            ep.startswith("gym.envs.atari")
            and not spec.id.startswith("Pong")
            and not spec.id.startswith("Seaquest")
        )
    ):
        return True
    return False


@pytest.fixture(scope="module", autouse=True)
def envs():
    all_envs = []
    env_map = {}
    lg = getLogger(__file__)
    # Find the newest version of non-skippable environments.
    for env_raw_name, env_spec in gym.envs.registry.env_specs.items():
        if not should_skip(env_spec):
            env_name, env_version = env_raw_name.split("-v")
            if env_name not in env_version or int(env_version) > env_map[env_name]:
                env_map[env_name] = int(env_version)

    # Create environments.
    for env_name, env_version in env_map.items():
        env_name = env_name + "-v" + str(env_version)
        lg.info(f"OpenAI gym {env_name} added")
        all_envs.append([lambda *_: gym.make(env_name) for _ in range(ENV_NUM)])
    lg.info("{} OpenAI gym environments to be tested.".format(len(all_envs)))
    return all_envs


class TestParallelWrapperDummy:
    ########################################################################
    # Test for ParallelWrapperDummy.reset
    ########################################################################
    param_test_reset = [
        (None, ENV_NUM),
        (choice(range(ENV_NUM)), 1),
        (sample(range(ENV_NUM), SAMPLE_NUM), SAMPLE_NUM),
        ([_ for _ in range(ENV_NUM)], ENV_NUM),
    ]

    @pytest.mark.parametrize("idx,reset_num", param_test_reset)
    def test_reset(self, envs, idx, reset_num):
        for env_list in envs:
            dummy_wrapper = openai_gym.ParallelWrapperDummy(env_list)
            obsrvs = dummy_wrapper.reset(idx)
            dummy_wrapper.close()

            assert len(obsrvs) == reset_num
            for obsrv in obsrvs:
                assert dummy_wrapper.observation_space.contains(
                    obsrv
                ), "Required observation form: {}, Actual observation: {}".format(
                    str(dummy_wrapper.observation_space), obsrv
                )

    ########################################################################
    # Test for ParallelWrapperDummy.step
    ########################################################################
    param_test_step = [
        (None, ENV_NUM),
        (choice(range(ENV_NUM)), 1),
        (sample(range(ENV_NUM), SAMPLE_NUM), SAMPLE_NUM),
        ([_ for _ in range(ENV_NUM)], ENV_NUM),
    ]

    @pytest.mark.parametrize("idx,act_num", param_test_step)
    def test_step(self, envs, idx, act_num):
        for env_list in envs:
            dummy_wrapper = openai_gym.ParallelWrapperDummy(env_list)
            action = [mock_action(dummy_wrapper.action_space) for _ in range(act_num)]
            dummy_wrapper.reset(idx)
            obsrvs, reward, terminal, info = dummy_wrapper.step(action, idx)
            dummy_wrapper.close()

            assert len(obsrvs) == act_num
            assert len(reward) == act_num
            assert len(terminal) == act_num
            assert len(info) == act_num and isinstance(info[0], dict)
            for obsrv in obsrvs:
                assert dummy_wrapper.observation_space.contains(
                    obsrv
                ), "Required observation form: {}, Actual observation: {}".format(
                    str(dummy_wrapper.observation_space), obsrv
                )

    ########################################################################
    # Test for ParallelWrapperDummy.seed
    ########################################################################
    param_test_seed = [
        None,
        choice(range(ENV_NUM)),
        sample(range(ENV_NUM), SAMPLE_NUM),
        [_ for _ in range(ENV_NUM)],
    ]

    @pytest.mark.parametrize("idx", param_test_seed)
    def test_seed(self, envs, idx):
        for env_list in envs:
            dummy_wrapper = openai_gym.ParallelWrapperDummy(env_list)
            seeds = dummy_wrapper.seed()
            dummy_wrapper.close()
            assert len(seeds) == ENV_NUM

    ########################################################################
    # Test for ParallelWrapperDummy.render
    ########################################################################
    param_test_render = [
        (None, ENV_NUM),
        (choice(range(ENV_NUM)), 1),
        (sample(range(ENV_NUM), SAMPLE_NUM), SAMPLE_NUM),
        ([_ for _ in range(ENV_NUM)], ENV_NUM),
    ]

    @pytest.mark.parametrize("idx,render_num", param_test_render)
    def test_render(self, envs, idx, render_num):
        for env_list in envs:
            dummy_wrapper = openai_gym.ParallelWrapperDummy(env_list)
            dummy_wrapper.reset(idx)
            rendered = dummy_wrapper.render(idx)
            dummy_wrapper.close()
            assert len(rendered) == render_num
            assert isinstance(rendered[0], np.ndarray)
            assert rendered[0].ndim == 3 and rendered[0].shape[-1] == 3

    ########################################################################
    # Test for ParallelWrapperDummy.close
    ########################################################################
    def test_close(self, envs):
        for env_list in envs:
            dummy_wrapper = openai_gym.ParallelWrapperDummy(env_list)
            dummy_wrapper.close()

    ########################################################################
    # Test for ParallelWrapperDummy.active
    ########################################################################
    def test_active(self, envs):
        for env_list in envs:
            dummy_wrapper = openai_gym.ParallelWrapperDummy(env_list)
            dummy_wrapper.reset()
            active = dummy_wrapper.active()
            dummy_wrapper.close()
            assert len(active) == ENV_NUM

    ########################################################################
    # Test for ParallelWrapperDummy.size
    ########################################################################
    def test_size(self, envs):
        dummy_wrapper = openai_gym.ParallelWrapperDummy(envs[0])
        assert dummy_wrapper.size() == ENV_NUM
        dummy_wrapper.close()


class TestParallelWrapperSubProc:
    ########################################################################
    # Test for ParallelWrapperSubProc.reset
    ########################################################################
    param_test_reset = [
        (None, ENV_NUM),
        (choice(range(ENV_NUM)), 1),
        (sample(range(ENV_NUM), SAMPLE_NUM), SAMPLE_NUM),
        ([_ for _ in range(ENV_NUM)], ENV_NUM),
    ]

    @pytest.mark.parametrize("idx,reset_num", param_test_reset)
    def test_reset(self, envs, idx, reset_num):
        for env_list in envs:
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(env_list)
            obsrvs = subproc_wrapper.reset(idx)
            subproc_wrapper.close()

            assert len(obsrvs) == reset_num
            for obsrv in obsrvs:
                assert subproc_wrapper.observation_space.contains(
                    obsrv
                ), "Required observation form: {}, Actual observation: {}".format(
                    str(subproc_wrapper.observation_space), obsrv
                )

    ########################################################################
    # Test for ParallelWrapperSubProc.step
    ########################################################################
    param_test_step = [
        (None, ENV_NUM),
        (choice(range(ENV_NUM)), 1),
        (sample(range(ENV_NUM), SAMPLE_NUM), SAMPLE_NUM),
        ([_ for _ in range(ENV_NUM)], ENV_NUM),
    ]

    @pytest.mark.parametrize("idx,act_num", param_test_step)
    def test_step(self, envs, idx, act_num):
        for env_list in envs:
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(env_list)
            action = [mock_action(subproc_wrapper.action_space) for _ in range(act_num)]
            subproc_wrapper.reset(idx)
            obsrvs, reward, terminal, info = subproc_wrapper.step(action, idx)
            subproc_wrapper.close()

            assert len(obsrvs) == act_num
            assert len(reward) == act_num
            assert len(terminal) == act_num
            assert len(info) == act_num and isinstance(info[0], dict)
            for obsrv in obsrvs:
                assert subproc_wrapper.observation_space.contains(
                    obsrv
                ), "Required observation form: {}, Actual observation: {}".format(
                    str(subproc_wrapper.observation_space), obsrv
                )

    ########################################################################
    # Test for ParallelWrapperSubProc.seed
    ########################################################################
    param_test_seed = [
        None,
        choice(range(ENV_NUM)),
        sample(range(ENV_NUM), SAMPLE_NUM),
        [_ for _ in range(ENV_NUM)],
    ]

    @pytest.mark.parametrize("idx", param_test_seed)
    def test_seed(self, envs, idx):
        for env_list in envs:
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(env_list)
            seeds = subproc_wrapper.seed()
            subproc_wrapper.close()
            assert len(seeds) == ENV_NUM

    ########################################################################
    # Test for ParallelWrapperSubProc.render
    ########################################################################
    param_test_render = [
        (None, ENV_NUM),
        (choice(range(ENV_NUM)), 1),
        (sample(range(ENV_NUM), SAMPLE_NUM), SAMPLE_NUM),
        ([_ for _ in range(ENV_NUM)], ENV_NUM),
    ]

    @pytest.mark.parametrize("idx,render_num", param_test_render)
    def test_render(self, envs, idx, render_num):
        for env_list in envs:
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(env_list)
            subproc_wrapper.reset(idx)
            rendered = subproc_wrapper.render(idx)
            subproc_wrapper.close()
            assert len(rendered) == render_num
            assert isinstance(rendered[0], np.ndarray)
            assert rendered[0].ndim == 3 and rendered[0].shape[-1] == 3

    ########################################################################
    # Test for ParallelWrapperSubProc.close
    ########################################################################
    def test_close(self, envs):
        for env_list in envs:
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(env_list)
            subproc_wrapper.close()

    ########################################################################
    # Test for ParallelWrapperSubProc.active
    ########################################################################
    def test_active(self, envs):
        for env_list in envs:
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(env_list)
            subproc_wrapper.reset()
            active = subproc_wrapper.active()
            subproc_wrapper.close()
            assert len(active) == ENV_NUM

    def test_size(self, envs):
        subproc_wrapper = openai_gym.ParallelWrapperSubProc(envs[0])
        assert subproc_wrapper.size() == ENV_NUM
        subproc_wrapper.close()
