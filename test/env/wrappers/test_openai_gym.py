"""
Currently, openai gym offers a great range of environments and we try to
test all kinds of them (not all), however, "robotics" and "mujoco"
requires a license and we cannot test them, but theoretically they
should work just fine.

Submit us a issue if you have found any problem.
"""
from random import choice, sample
from machin.env.wrappers import openai_gym
from machin.utils.logging import default_logger
from test.util_platforms import linux_only_forall

import pytest
import gym
import numpy as np

linux_only_forall()
ENV_NUM = 2
SAMPLE_NUM = 2
WORKER_NUM = 2


def mock_action(action_space: gym.spaces.Space):
    return action_space.sample()


@pytest.fixture(scope="module", autouse=True)
def envs():
    names = ["CartPole-v0"]
    creators = []

    # Create environments.
    for name in names:
        creators.append([lambda *_: gym.make(name) for _ in range(ENV_NUM)])
    return names, creators


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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            dummy_wrapper = openai_gym.ParallelWrapperDummy(creators)
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            dummy_wrapper = openai_gym.ParallelWrapperDummy(creators)
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            dummy_wrapper = openai_gym.ParallelWrapperDummy(creators)
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            dummy_wrapper = openai_gym.ParallelWrapperDummy(creators)
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            dummy_wrapper = openai_gym.ParallelWrapperDummy(creators)
            dummy_wrapper.close()

    ########################################################################
    # Test for ParallelWrapperDummy.active
    ########################################################################
    def test_active(self, envs):
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            dummy_wrapper = openai_gym.ParallelWrapperDummy(creators)
            dummy_wrapper.reset()
            active = dummy_wrapper.active()
            dummy_wrapper.close()
            assert len(active) == ENV_NUM

    ########################################################################
    # Test for ParallelWrapperDummy.size
    ########################################################################
    def test_size(self, envs):
        dummy_wrapper = openai_gym.ParallelWrapperDummy(envs[1][0])
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(creators)
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(creators)
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(creators)
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(creators)
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
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(creators)
            subproc_wrapper.close()

    ########################################################################
    # Test for ParallelWrapperSubProc.active
    ########################################################################
    def test_active(self, envs):
        for name, creators in zip(*envs):
            default_logger.info(f"Testing on env {name}")
            subproc_wrapper = openai_gym.ParallelWrapperSubProc(creators)
            subproc_wrapper.reset()
            active = subproc_wrapper.active()
            subproc_wrapper.close()
            assert len(active) == ENV_NUM

    def test_size(self, envs):
        subproc_wrapper = openai_gym.ParallelWrapperSubProc(envs[1][0])
        assert subproc_wrapper.size() == ENV_NUM
        subproc_wrapper.close()
