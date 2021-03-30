from machin.frame.noise.generator import (
    NormalNoiseGen,
    UniformNoiseGen,
    ClippedNormalNoiseGen,
    OrnsteinUhlenbeckNoiseGen,
)

import torch as t


class TestAllNoiseGen(object):
    ########################################################################
    # Test for all noise generators
    ########################################################################
    def test_normal_noise_gen(self, pytestconfig):
        noise_gen = NormalNoiseGen([1, 2])
        noise_gen()
        noise_gen(pytestconfig.getoption("gpu_device"))
        str(noise_gen)

    def test_clipped_normal_noise_gen(self, pytestconfig):
        noise_gen = ClippedNormalNoiseGen([1, 2])
        noise_gen()
        noise_gen(pytestconfig.getoption("gpu_device"))
        str(noise_gen)

    def test_uniform_noise_gen(self, pytestconfig):
        noise_gen = UniformNoiseGen([1, 2])
        noise_gen()
        noise_gen(pytestconfig.getoption("gpu_device"))
        str(noise_gen)

    def test_ou_noise_gen(self, pytestconfig):
        noise_gen = OrnsteinUhlenbeckNoiseGen([1, 2])
        noise_gen2 = OrnsteinUhlenbeckNoiseGen([1, 2], x0=t.ones([1, 2]))
        noise_gen()
        noise_gen.reset()
        noise_gen2.reset()
        noise_gen(pytestconfig.getoption("gpu_device"))
        str(noise_gen)
