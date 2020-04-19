import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

from utils.helper_classes import Object


# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

class Agent(Object):
    def __init__(self):
        super(Agent, self).__init__()
        self.data = {"hull": None, "legs": None, "joints": None, "lidar": None,
                     "is_carrying": False}

    def attr(self, item, change=None):
        if change is not None:
            self.data[item] = change[0]
        else:
            return self.data[item]

    def reset(self):
        self.data = {"hull": None, "legs": None, "joints": None, "lidar": None,
                     "is_carrying": False}

    def get_fixtures(self):
        return [self.data["hull"]] + self.data["legs"]


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def begin_contact(self, A, B):
        if A.userData is None or B.userData is None:
            raise RuntimeError("Userdata not set")
        # agent hull should not contact anything, except the cargo it carries
        if A.userData.get("type", None) == "hull":
            if B.userData.get("type", None) == "cargo":
                agent = A.userData["agent"]
                self.env.agents[agent].is_carrying = True
            else:
                # hull contacted ground, game over
                self.env.game_over = True
        elif A.userData.get("type", None) == "leg":
            # lower/upper leg contacted ground
            agent = A.userData["agent"]
            category_num = A.userData["category_num"]
            self.env.agents[agent].legs[category_num].ground_contact = True
        elif A.userData.get("type", None) == "cargo":
            if B.userData.get("type", None) == "ground":
                self.env.game_over = True

    def end_contact(self, A, B):
        if A.userData is None or B.userData is None:
            raise RuntimeError("Userdata not set")
        # agent hull should not contact anything, except the cargo it carries
        if A.userData.get("type", None) == "hull":
            if B.userData.get("type", None) == "cargo":
                agent = A.userData["agent"]
                self.env.agents[agent].is_carrying = False
        elif A.userData.get("type", None) == "leg":
            # lower/upper leg contacted ground
            agent = A.userData["agent"]
            category_num = A.userData["category_num"]
            self.env.agents[agent].legs[category_num].ground_contact = False

    def BeginContact(self, contact):
        self.begin_contact(contact.fixtureA.body, contact.fixtureB.body)
        self.begin_contact(contact.fixtureB.body, contact.fixtureA.body)

    def EndContact(self, contact):
        self.end_contact(contact.fixtureA.body, contact.fixtureB.body)
        self.end_contact(contact.fixtureB.body, contact.fixtureA.body)


class LidarCallback(Box2D.b2.rayCastCallback):
    def __init__(self):
        super(LidarCallback, self).__init__()
        self.p2 = None
        self.fraction = None

    def ReportFixture(self, fixture, point, normal, fraction):
        # if not ground, do not report
        if (fixture.filterData.categoryBits & 1) == 0:
            return -1
        self.p2 = point
        self.fraction = fraction
        return fraction


class BipedalMultiWalker(gym.Env, EzPickle):
    FPS = 50
    SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
    VIEWPORT_W = 600
    VIEWPORT_H = 400
    INITIAL_RANDOM = 5

    MOTORS_TORQUE = 80
    SPEED_HIP = 4
    SPEED_KNEE = 6
    LIDAR_RANGE = 160 / SCALE
    LIDAR_RESOLUTION = 10  # 10 samples per half loop
    LIDAR_DELAY = 0  # times of a single loop time

    HULL_POLY = [
        (-30, +9), (+6, +9), (+34, +1),
        (+34, -8), (-30, -8)
    ]
    LEG_DOWN = -8 / SCALE
    LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

    TERRAIN_STEP = 14 / SCALE
    TERRAIN_LENGTH_AFTER_STARTPAD = 200  # in steps
    TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
    TERRAIN_GRASS = 10  # low long are grass spots, in steps
    TERRAIN_STARTPAD_PER_AGENT = 15  # in steps
    TERRAIN_AGENT_DISTANCE = 8  # in steps

    CARGO_HEIGHT = 4 / SCALE
    CARGO_Y_OFFSET = 9 / SCALE  # same as max y of hull poly

    MAX_MOVE_REWARD = 300
    MAX_CARRY_REWARD = 300
    GAME_OVER_PUNISH = 200
    NOT_CARRYING_PUNISH = 100

    FRICTION = 2.5

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    hardcore = False

    def __init__(self, agent_num=1):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = []
        self.cargo = None
        self.agents = [Agent() for i in range(agent_num)]
        self.agent_num = agent_num
        self.draw_list = []
        self.game_over = False
        self.scroll = 0.0
        self.lidar_step = 0

        # culmulative reward, shaping-prev_shaping = increased reward (or step reward)
        self.prev_sum_reward = None

        # fd_polygon and fd_edge are used in terrain generation
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            friction=self.FRICTION
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=
                            [(0, 0),
                             (1, 1)]),
            friction=self.FRICTION,
            categoryBits=0x0001,
        )

        self.TERRAIN_STARTPAD_LENGTH = counter = self.TERRAIN_STARTPAD_PER_AGENT * self.agent_num
        self.TERRAIN_LENGTH = self.TERRAIN_STARTPAD_LENGTH + self.TERRAIN_LENGTH_AFTER_STARTPAD

        self.reset()

        action_range = np.array([1] * (4 * agent_num))
        observe_range = np.array([np.inf] * (24 * agent_num))
        # define action space and observation space in gym.Env
        self.action_space = spaces.Box(-action_range, action_range, dtype=np.float32)
        self.observation_space = spaces.Box(-observe_range, observe_range, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []

        self.world.DestroyBody(self.cargo)
        self.cargo = None
        self.cargo_init_pos = None

        for a in self.agents:
            self.world.DestroyBody(a.hull)
            a.hull = None
            for leg in a.legs:
                self.world.DestroyBody(leg)
            a.legs = []
            # joints are automatically destroyed if attached to a body
            a.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = self.TERRAIN_HEIGHT
        # only generate one of STUMP, STAIRS, PIT
        oneshot = False
        counter = self.TERRAIN_STARTPAD_LENGTH
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(self.TERRAIN_LENGTH):
            x = i * self.TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(self.TERRAIN_HEIGHT - y)
                if i > self.TERRAIN_STARTPAD_LENGTH:
                    velocity += self.np_random.uniform(-1, 1) / self.SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x, y),
                    (x + self.TERRAIN_STEP, y),
                    (x + self.TERRAIN_STEP, y - 4 * self.TERRAIN_STEP),
                    (x, y - 4 * self.TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon,
                    userData={"type": "ground"}
                )
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [(p[0] + self.TERRAIN_STEP * counter, p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon,
                    userData={"type": "ground"}
                )
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * self.TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x, y),
                    (x + counter * self.TERRAIN_STEP, y),
                    (x + counter * self.TERRAIN_STEP, y + counter * self.TERRAIN_STEP),
                    (x, y + counter * self.TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon,
                    userData={"type": "ground"}
                )
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x + (s * stair_width) * self.TERRAIN_STEP,
                         y + (s * stair_height) * self.TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * self.TERRAIN_STEP,
                         y + (s * stair_height) * self.TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * self.TERRAIN_STEP,
                         y + (-1 + s * stair_height) * self.TERRAIN_STEP),
                        (x + (s * stair_width) * self.TERRAIN_STEP,
                         y + (-1 + s * stair_height) * self.TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon,
                        userData={"type": "ground"}
                    )
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * self.TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                # a random length of grass patch
                counter = self.np_random.randint(self.TERRAIN_GRASS / 2, self.TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(self.TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge,
                userData={"type": "ground"}
            )
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(self.TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, self.TERRAIN_LENGTH) * self.TERRAIN_STEP
            y = self.VIEWPORT_H / self.SCALE * 3 / 4
            poly = [
                (x + 15 * self.TERRAIN_STEP * math.sin(3.14 * 2 * a / 5) +
                 self.np_random.uniform(0, 5 * self.TERRAIN_STEP),
                 y + 5 * self.TERRAIN_STEP * math.cos(3.14 * 2 * a / 5) +
                 self.np_random.uniform(0, 5 * self.TERRAIN_STEP))
                for a in range(5)]
            x1 = min([p[0] for p in poly])
            x2 = max([p[0] for p in poly])
            self.cloud_poly.append((poly, x1, x2))

    def _generate_agents(self):
        begin_x = self.TERRAIN_STEP * self.TERRAIN_STARTPAD_PER_AGENT / 2
        for i in range(self.agent_num):
            init_x = begin_x + self.TERRAIN_STEP * self.TERRAIN_AGENT_DISTANCE * i
            init_y = self.TERRAIN_HEIGHT + 2 * self.LEG_H
            agent = self.agents[i]
            agent.init_pos = (init_x, init_y)
            agent.lidar = [LidarCallback() for _ in range(self.LIDAR_RESOLUTION)]

            agent.hull = self.world.CreateDynamicBody(
                position=(init_x, init_y),
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in self.HULL_POLY]),
                    density=5.0,
                    friction=0.1,
                    categoryBits=0x0100,
                    maskBits=0x0011,  # collide with ground and cargo
                    restitution=0.0,  # 0.99 bouncy
                ),
                userData={"agent": i, "type": "hull", "category_num": 0}
            )
            agent.hull.color1 = (0.5, 0.4, 0.9)
            agent.hull.color2 = (0.3, 0.3, 0.5)
            agent.hull.ApplyForceToCenter(
                (self.np_random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM), 0), True
            )

            agent.legs = []
            agent.joints = []
            for j in [-1, +1]:
                upper_leg = self.world.CreateDynamicBody(
                    position=(init_x, init_y - self.LEG_H / 2 - self.LEG_DOWN),
                    angle=(j * 0.05),
                    fixtures=fixtureDef(
                        shape=polygonShape(box=(self.LEG_W / 2, self.LEG_H / 2)),
                        density=1.0,
                        restitution=0.0,
                        categoryBits=0x0100,
                        maskBits=0x0001,  # collide with ground only
                    ),
                    userData={"agent": i, "type": "leg", "category_num": j + 1}
                )
                upper_leg.color1 = (0.6 - j / 10., 0.3 - j / 10., 0.5 - j / 10.)
                upper_leg.color2 = (0.4 - j / 10., 0.2 - j / 10., 0.3 - j / 10.)
                rjd = revoluteJointDef(
                    bodyA=agent.hull,
                    bodyB=upper_leg,
                    localAnchorA=(0, self.LEG_DOWN),
                    localAnchorB=(0, self.LEG_H / 2),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=self.MOTORS_TORQUE,
                    motorSpeed=j,
                    lowerAngle=-0.8,
                    upperAngle=1.1,
                )
                agent.legs.append(upper_leg)
                agent.joints.append(self.world.CreateJoint(rjd))

                lower_leg = self.world.CreateDynamicBody(
                    position=(init_x, init_y - self.LEG_H * 3 / 2 - self.LEG_DOWN),
                    angle=(j * 0.05),
                    fixtures=fixtureDef(
                        shape=polygonShape(box=(0.8 * self.LEG_W / 2, self.LEG_H / 2)),
                        density=1.0,
                        restitution=0.0,
                        categoryBits=0x0100,
                        maskBits=0x0001,  # collide with ground only
                    ),
                    userData={"agent": i, "type": "leg", "category_num": j + 2}
                )
                lower_leg.color1 = (0.6 - j / 10., 0.3 - j / 10., 0.5 - j / 10.)
                lower_leg.color2 = (0.4 - j / 10., 0.2 - j / 10., 0.3 - j / 10.)
                rjd2 = revoluteJointDef(
                    bodyA=upper_leg,
                    bodyB=lower_leg,
                    localAnchorA=(0, -self.LEG_H / 2),
                    localAnchorB=(0, self.LEG_H / 2),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=self.MOTORS_TORQUE,
                    motorSpeed=1,
                    lowerAngle=-1.6,
                    upperAngle=-0.1,
                )
                lower_leg.ground_contact = False
                agent.legs.append(lower_leg)
                agent.joints.append(self.world.CreateJoint(rjd2))

    def _generate_cargo(self):
        length = self.TERRAIN_STEP * self.TERRAIN_AGENT_DISTANCE * (self.agent_num - 0.5)
        init_x = self.TERRAIN_STEP * self.TERRAIN_STARTPAD_PER_AGENT / 2
        init_y = self.TERRAIN_HEIGHT + 2 * self.LEG_H + self.CARGO_Y_OFFSET
        self.cargo = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(box=(length, self.CARGO_HEIGHT)),
                density=1.0,
                restitution=0.0,
                categoryBits=0x0010,
                maskBits=0x0101,  # collide with ground and agents
            ),
            userData={"type": "cargo"}
        )
        self.cargo.color1 = (0.5, 0.4, 0.9)
        self.cargo.color2 = (0.3, 0.3, 0.5)
        self.cargo_init_pos = [init_x, init_y]

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_sum_reward = np.zeros(self.agent_num)
        self.scroll = 0.0
        self.lidar_step = 0

        W = self.VIEWPORT_W / self.SCALE
        H = self.VIEWPORT_H / self.SCALE

        for agent in self.agents:
            agent.reset()

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        self._generate_agents()
        self._generate_cargo()

        self.draw_list = self.terrain + [self.cargo]
        for agent in self.agents:
            self.draw_list += agent.get_fixtures()

        return self.step(np.array([0, 0, 0, 0] * self.agent_num))[0]

    def step(self, action):
        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well

        for i in range(self.agent_num):
            agent = self.agents[i]
            if control_speed:
                offset = i * 4
                agent.joints[0].motorSpeed = float(self.SPEED_HIP * np.clip(action[0 + offset], -1, 1))
                agent.joints[1].motorSpeed = float(self.SPEED_KNEE * np.clip(action[1 + offset], -1, 1))
                agent.joints[2].motorSpeed = float(self.SPEED_HIP * np.clip(action[2 + offset], -1, 1))
                agent.joints[3].motorSpeed = float(self.SPEED_KNEE * np.clip(action[3 + offset], -1, 1))
            else:
                offset = i * 4
                agent.joints[0].motorSpeed = float(self.SPEED_HIP * np.sign(action[0 + offset]))
                agent.joints[0].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[0 + offset]), 0, 1))
                agent.joints[1].motorSpeed = float(self.SPEED_KNEE * np.sign(action[1 + offset]))
                agent.joints[1].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[1 + offset]), 0, 1))
                agent.joints[2].motorSpeed = float(self.SPEED_HIP * np.sign(action[2 + offset]))
                agent.joints[2].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[2 + offset]), 0, 1))
                agent.joints[3].motorSpeed = float(self.SPEED_KNEE * np.sign(action[3 + offset]))
                agent.joints[3].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[3 + offset]), 0, 1))
        self.world.Step(1.0 / self.FPS, 6 * 30, 2 * 30)

        min_x = np.inf
        max_x = -np.inf
        state = []
        reward = np.zeros(self.agent_num)
        cargo_reward = self.MAX_MOVE_REWARD * (self.cargo.position[0] - self.cargo_init_pos[0])\
                       / (self.TERRAIN_LENGTH * self.TERRAIN_STEP)

        for i in range(self.agent_num):
            agent = self.agents[i]
            pos = agent.hull.position
            vel = agent.hull.linearVelocity
            min_x = min(pos[0], min_x)
            max_x = max(pos[0], max_x)

            for j in range(10):
                agent.lidar[j].fraction = 1.0
                agent.lidar[j].p1 = pos
                agent.lidar[j].p2 = (
                    pos[0] + math.sin(1.5 * j / 10.0) * self.LIDAR_RANGE,
                    pos[1] - math.cos(1.5 * j / 10.0) * self.LIDAR_RANGE)
                self.world.RayCast(agent.lidar[j], agent.lidar[j].p1, agent.lidar[j].p2)

            agent_state = [
                # Normal angles up to 0.5 here, but sure more is possible.
                agent.hull.angle,
                2.0 * agent.hull.angularVelocity / self.FPS,
                0.3 * vel.x * (self.VIEWPORT_W / self.SCALE) / self.FPS,  # Normalized to get -1..1 range
                0.3 * vel.y * (self.VIEWPORT_H / self.SCALE) / self.FPS,
                agent.joints[0].angle,
                # This will give 1.1 on high up, but it's still OK
                # (and there should be spikes on hitting the ground, that's normal too)
                agent.joints[0].speed / self.SPEED_HIP,
                agent.joints[1].angle + 1.0,
                agent.joints[1].speed / self.SPEED_KNEE,
                1.0 if agent.legs[1].ground_contact else 0.0,
                agent.joints[2].angle,
                agent.joints[2].speed / self.SPEED_HIP,
                agent.joints[3].angle + 1.0,
                agent.joints[3].speed / self.SPEED_KNEE,
                1.0 if agent.legs[3].ground_contact else 0.0
            ]
            agent_state += [l.fraction for l in agent.lidar]
            assert len(agent_state) == 24
            state += agent_state

            # moving forward is a way to receive reward
            sum_reward = self.MAX_MOVE_REWARD * (pos[0] - agent.init_pos[0]) \
                         / (self.TERRAIN_LENGTH * self.TERRAIN_STEP)

            # carrying cargo forward is also a way to receive reward
            sum_reward += cargo_reward

            # keep head straight
            sum_reward -= 5.0 * abs(state[0])

            # keep contact with cargo
            # if not agent.is_carrying:
            #     sum_reward -= self.NOT_CARRYING_PUNISH

            agent_reward = sum_reward - self.prev_sum_reward[i]
            self.prev_sum_reward[i] = sum_reward

            ## punishment for using power
            #for a in action:
            #    agent_reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            #    # normalized to about -50.0 using heuristic, more optimal agent should spend less
            reward[i] = agent_reward

        self.scroll = min_x - self.VIEWPORT_W / self.SCALE / 5
        is_finished = False
        if self.game_over or min_x < 0 or self.cargo.position[0] < 0:
            reward[:] = -self.GAME_OVER_PUNISH
            is_finished = True
        if self.cargo.position[0] > (self.TERRAIN_LENGTH - self.TERRAIN_GRASS) * self.TERRAIN_STEP:
            is_finished = True
        return np.array(state), reward, is_finished, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            if mode == "rgb_array":
                self.viewer.window.set_visible(False)
        self.viewer.set_bounds(self.scroll, self.VIEWPORT_W / self.SCALE + self.scroll,
                               0, self.VIEWPORT_H / self.SCALE)

        self.viewer.draw_polygon([
            (self.scroll, 0),
            (self.scroll + self.VIEWPORT_W / self.SCALE, 0),
            (self.scroll + self.VIEWPORT_W / self.SCALE, self.VIEWPORT_H / self.SCALE),
            (self.scroll, self.VIEWPORT_H / self.SCALE),
        ], color=(0.9, 0.9, 1.0))
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2: continue
            if x1 > self.scroll / 2 + self.VIEWPORT_W / self.SCALE: continue
            self.viewer.draw_polygon([(p[0] + self.scroll / 2, p[1]) for p in poly], color=(1, 1, 1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + self.VIEWPORT_W / self.SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        # lidar scans in a reciprocating manner
        # so 2 * len(self.lidar) completes a loop from 0 -> last -> 0 messure point
        self.lidar_step = (self.lidar_step + 1) % (self.LIDAR_RESOLUTION * 2 * (1 + self.LIDAR_DELAY))
        i = self.lidar_step

        if i < 2 * self.LIDAR_RESOLUTION:
            for agent in self.agents:
                l = agent.lidar[i] if i < self.LIDAR_RESOLUTION else agent.lidar[self.LIDAR_RESOLUTION - i - 1]
                self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        for obj in self.draw_list:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = self.TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / self.SCALE
        x = self.TERRAIN_STEP * 3
        self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2)
        f = [(x, flagy2), (x, flagy2 - 10 / self.SCALE), (x + 25 / self.SCALE, flagy2 - 5 / self.SCALE)]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class BipedalMultiWalkerHardcore(BipedalMultiWalker):
    hardcore = True