import importlib
import json
import math
import os.path
import random

import numpy as np

from gym.envs.classic_control import rendering

from envs.DefenceAgent import DefenceAgent
from envs.J20 import J20
from envs.Radar import Radar
from envs.Agent import Agent

import reward_weight.weight01


class Swarm(object):
    """
    定义环境中的所有智能体，及其状态转移方式
    """

    def __init__(self, agent_num, args_all):
        self.args_all = args_all
        radar1 = Radar(radar_id='radar1', x=0e3, y=0e3) #head=([-1,1])[random.randint(0,1)]
        radar2 = Radar(radar_id='radar2', x=0e3, y=10e3) #head=([-1,1])[random.randint(0,1)]
        radar3 = Radar(radar_id='radar3', x=0e3, y=-10e3)
        self.radars = [radar1, radar2, radar3]
        j20_1 = J20(j20_id=0, x=-300e3, y=0, target=[0, 0])
        j20_2 = J20(j20_id=1, x=-300e3, y=10e3, target=[0, 10e3])
        j20_3 = J20(j20_id=2, x=-300e3, y=-10e3, target=[0, -10e3])
        # self.J20 = J20(x=0, y=0, target=[150e3, 0]

        self.J20s = [j20_1, j20_2, j20_3]
        pos = [
            [-60e3, 2e3],
            [-60e3, -2e3],
            [-60e3, 3e3],
            [-60e3, -3e3],

            [-80e3, 12e3],
            [-80e3, 11e3],
            [-80e3, 10e3],
            [-80e3, 9e3],

            [-100e3, -12e3],
            [-100e3, -11e3],
            [-100e3, -10e3],
            [-100e3, -9e3],
        ]  # agent的初始化位置
        plane_num = len(pos)

        pos_defence = [
            [-40e3, 0],
            [-50e3, 10e3],
            [-60e3, -10e3],
        ]


        self.agents = [Agent(
            agent_id=agent_id,
            init_pos=pos[agent_id],
            env=self,
            reward_weight=self.args_all.reward_weight,
            J_20=self.J20s[agent_id//4],
        ) for agent_id in range(plane_num)]
        self.defence_agents = [DefenceAgent(
            agent_id=agent_id,
            init_pos=pos_defence[agent_id],
            env=self,
            reward_weight=self.args_all.reward_weight,

        ) for agent_id in range(3)]
        self.agent_num = agent_num

        self.radar_allocate()  # 为无人机分配雷达目标
        self.cover_allocate()  # 为无人机分配我方掩护目标

        self.itr = 0  # 迭代步数控制
        self.save_dir = None  # 存储轨迹json文件的文件夹路径
        self.make_dir()
        self.file_num = len(os.listdir(self.save_dir))
        self.trace_file = f"{self.save_dir}/run{self.file_num}.json"

        self.dic = {
            'J20': {},
            'agents': {},
            'radars': {},
            'defence_agents': {},
        }  # 用于存储各个实体（雷达、无人机等）的轨迹，用于后续画图

        # 以下两个for循环用于完成对self.dic的agents和radars部分的初始化
        for agent in self.agents:
            temp = {
                'x': [],
                'y': [],
                'heading': [],
                'reward': [],
                'hp': [],
            }
            self.dic['agents'][agent.agent_id] = temp
        for j20 in self.J20s:
            temp = {
                'x': [],
                'y': [],
                'cover_rates': [],
            }
            self.dic['J20'][j20.j20_id] = temp
        for radar in self.radars:
            self.dic['radars'][radar.radar_id] = {}
            self.dic['radars'][radar.radar_id]['x'] = []
            self.dic['radars'][radar.radar_id]['y'] = []
        for defence_agent in self.defence_agents:
            temp = {
                'x': [],
                'y': [],
                'reward': [],
                'attack_x': [],
                'attack_y': [],
                'hp': [],
            }
            self.dic['defence_agents'][defence_agent.agent_id] = temp

    def radar_allocate(self):
        for agent in self.agents:
            agent.radar = self.radars[agent.agent_id // 4]
            agent.radar.agents.append(agent)

    def cover_allocate(self):
        for agent in self.agents:
            agent.J20 = self.J20s[0]

    def update(self, actions, defence_actions):
        """
        actions: n个agent的动作的集合，actions[i]为第i+1个agent的动作向量，
        actions的形状为agent_num行，action_dim列
        """
        self.itr += 1

        for j20 in self.J20s:
            j20.update()
        for radar in self.radars:
            radar.update()

        for i, agent in enumerate(self.agents):
            agent.update(actions[i])
        for i, agent in enumerate(self.defence_agents):
            agent.update(defence_actions[i])
        if self.args_all.use_render:
            self.log_update()

        # 当我方飞机飞行到一定距离后，重置整个环境，不等式右边目前只是用于测试功能，后续可以进行修改
        # if self.J20.x >= self.radars[0].x - 40000:
        #     self.reset()

    def log_update(self):
        for agent in self.agents:
            self.dic['agents'][agent.agent_id]['x'].append(agent.x)
            self.dic['agents'][agent.agent_id]['y'].append(agent.y)
            self.dic['agents'][agent.agent_id]['heading'].append(agent.heading)
            self.dic['agents'][agent.agent_id]['hp'].append(agent.hp)
        # for itr, radar in enumerate(self.radars):
        #     self.dic['radars'][radar.radar_id]['x'].append(radar.x)
        #     self.dic['radars'][radar.radar_id]['y'].append(radar.y)
        #
        # for itr, j20 in enumerate(self.J20s):
        #     self.dic['J20'][itr]['x'].append(j20.x)
        #     self.dic['J20'][itr]['y'].append(j20.y)

        for agent in self.defence_agents:
            self.dic['defence_agents'][agent.agent_id]['x'].append(agent.x)
            self.dic['defence_agents'][agent.agent_id]['y'].append(agent.y)
            self.dic['defence_agents'][agent.agent_id]['reward'].append(agent.reward)
            self.dic['defence_agents'][agent.agent_id]['attack_x'].append(agent.attack_position[0])
            self.dic['defence_agents'][agent.agent_id]['attack_y'].append(agent.attack_position[1])
            self.dic['defence_agents'][agent.agent_id]['hp'].append(agent.hp)


        if self.args_all.use_render:
            if self.itr + 1 == self.args_all.max_step:
                with open(self.trace_file, 'w') as fo:
                    json.dump(self.dic, fo, indent=4)
        else:
            pass
            # if self.itr % 200 == 0:
            #     with open(self.trace_file, 'w') as fo:
            #         json.dump(self.dic, fo, indent=4)

    def make_dir(self):
        from pathlib import Path
        if self.args_all.use_render:
            self.save_dir = Path(
                os.path.abspath("./") + "/trace"
            ) / self.args_all.scenario_name / "render"
        else:
            self.save_dir = Path(
                os.path.abspath("./") + "/trace"
            ) / self.args_all.scenario_name / "train"
        if not self.save_dir.exists():
            os.makedirs(str(self.save_dir))

    def reset(self):
        self.__init__(agent_num=len(self.agents), args_all=self.args_all)


class MyEnv(object):
    """
    调用Swarm类，与light-mappo作者提供的环境接口进行交互
    """

    def __init__(self, args_all):
        # with open(os.path.abspath(f"./reward_weight/{args_all.scenario_name}.json"), 'r') as fo:
        #     temp = json.load(fo)
        args_all.reward_weight = reward_weight.weight01
        self.agent_num = args_all.num_agents  # 设置智能体(小飞机)的个数
        self.swarm = Swarm(agent_num=self.agent_num, args_all=args_all)
        self.obs_dim = len(self.swarm.defence_agents[0].obs)  # 设置智能体的观测维度
        self.action_dim = self.swarm.defence_agents[0].action_dim  # 设置智能体的动作维度
        self.viewer = None

    def reset(self):
        self.swarm.reset()
        sub_agent_obs = []
        for ag in self.swarm.agents:
            sub_agent_obs.append(np.array(ag.obs))

        sub_defence_agent_obs = []
        for ag in self.swarm.defence_agents:
            sub_defence_agent_obs.append(np.array(ag.obs))
        return sub_agent_obs, sub_defence_agent_obs

    def step(self, actions, defence_actions):
        sub_agent_obs = []
        sub_plane_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        self.swarm.update(actions=actions, defence_actions=defence_actions)
        for agent in self.swarm.agents:
            sub_plane_obs.append(np.array(agent.obs))

        for agent in self.swarm.defence_agents:
            sub_agent_obs.append(np.array(agent.obs))
            sub_agent_reward.append(agent.reward)
            sub_agent_done.append(agent.is_done())
            sub_agent_info.append({})

        return [sub_plane_obs,sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def render(self, mode='human'):
        screen_width = 1000
        screen_height = 1000
        scal = 200.0
        Y_add = 100000.0
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        # 雷达位置渲染
        for radar in self.swarm.radars:
            radar_point = rendering.make_circle(5)
            radar_point.set_color(10, 0, 0)
            position = [radar.x / scal, (radar.y + Y_add) / scal]
            radar_transform = rendering.Transform(translation=position)
            radar_point.add_attr(radar_transform)
            self.viewer.add_onetime(radar_point)

        # 目的地渲染
        target_point = rendering.make_circle(5)
        target_point.set_color(100, 0, 0)
        target_transform = rendering.Transform(
            translation=(self.swarm.J20.target[0] / scal, (self.swarm.J20.target[1] + Y_add) / scal))
        target_point.add_attr(target_transform)
        self.viewer.add_geom(target_point)

        # 掩护J20渲染
        guard_point = rendering.make_circle(5)
        guard_point.set_color(0, 0, 100)
        position = [self.swarm.J20.x / scal, (self.swarm.J20.y + Y_add) / scal]
        guard_transform = rendering.Transform(translation=position)
        guard_point.add_attr(guard_transform)
        self.viewer.add_onetime(guard_point)

        # 无人机渲染
        for ag in self.swarm.agents:
            UAV_point = rendering.make_circle(4)
            UAV_point.set_color(0, 100, 0)
            position = [ag.x / scal, (ag.y + Y_add) / scal]
            UAV_transform = rendering.Transform(translation=position)
            UAV_point.add_attr(UAV_transform)
            self.viewer.add_onetime(UAV_point)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
