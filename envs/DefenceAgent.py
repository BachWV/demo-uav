import numpy as np

from utils.util import merge_intervals
from envs.Radar import Radar
from envs.J20 import J20
class DefenceAgent(object):
    """
    我方激光
    """

    def __init__(self, agent_id: int, init_pos: [int, int], env, reward_weight: dict,):
        self.env = env
        self.agent_id = agent_id
        self.x, self.y = init_pos[0], init_pos[1]

        self.Round = 10_000 # 攻击范围
        self.attack_position = [0, 0] # 攻击目标
        self.attack_index = 0 # 攻击目标的索引
        self.attack_heading = 0 # 攻击目标的角度
        self.avail = 1 # 是否可用

        #攻击区域 attack_area = [x_min, x_max, y_min, y_max]
        self.attack_area = [0, 0, 0, 0]
        self.action_dim = 12

        self.reward_weight = reward_weight
        self.obs = list()
        self.get_obs()
        self.reward = 0
        self.hp = 100

    def get_obs(self):
        """
        获得当前agent对环境的观测 10维
        """
        distance = []
        for agent in self.env.agents:
            d = np.linalg.norm([agent.x - self.x, agent.y - self.y])
            if d < self.Round:
                distance.append(d/100)
            else:
                distance.append(100000)
        hp = [agent.hp for agent in self.env.agents]
        self.obs = distance + hp


    def update(self, action):
        if self.hp > 0:
            self.attack(action=action)
        self.get_obs()

    def attack(self, action):
        """

        """
        ai_idx = np.argsort(action)[::-1]
        distance = []
        rate = []


        for agent in self.env.agents:
            d = np.linalg.norm([agent.x - self.x, agent.y - self.y])
            r = 1 - agent.hp / 100
            r += 1 - d / self.Round
            rate.append(r)
            if d < self.Round:
                distance.append(d)
            else:
                distance.append(100000)
        # 查找distance中最小值对应的索引


        # index为排序后的从大到小的索引
        human_index = np.argsort(rate)[::-1]

        self.attack_position = [self.x, self.y]
        human = False
        if human:
            for ind in human_index:

                if distance[ind] < self.Round and self.env.agents[ind].hp > 0:
                    K = 1 - distance[ind] / self.Round
                    if ai_idx[0] == ind:
                        self.reward = 10
                    elif ai_idx[1] == ind:
                        self.reward = 5
                    elif ai_idx[2] == ind:
                        self.reward = 3
                    else:
                        self.reward = 0
                    self.env.agents[ind].hp_decrease(8 * K) # 将对应的agent的hp减少10
                    self.attack_position = [self.env.agents[ind].x, self.env.agents[ind].y]
                    break
        else:
            for ind in ai_idx:
                self.reward = 0
                if distance[ind] < self.Round and self.env.agents[ind].hp > 0:
                    K = 1 - distance[ind] / self.Round
                    self.env.agents[ind].hp -= 8 * K  # 将对应的agent的hp减少10
                    if self.env.agents[ind].hp < 0:
                        self.reward = 10
                    else:
                        self.reward = 0
                    self.attack_position = [self.env.agents[ind].x, self.env.agents[ind].y]
                    break

        assert self.reward <= 10, "奖励值超过上限"

    def hp_decrease(self, damage):
        self.hp -= damage


    def is_done(self) -> bool:
        """
        回合结束判定，如果x方向无人机位置超过了雷达位置，就结束当前回合
        """
        # if self.x > self.radar.x:
        #     return True
        return False


    def __rapo_rwd(self):
        """
        航点到达奖励，规定时间内到达进行奖励，否则惩罚，具体数值待定
        """
        return 0

    def __space_limit_punishment_y(self):
        """
        用来惩罚不必要的y轴方向探索空间
        """
        return 0

    def __space_limit_punishment_x(self):
        """
        用来惩罚uav倒飞
        """
        return 0

    def __reto_motion_1(self) -> int:
        """
        在这个场景中，我们并不需要三点一线，而是希望尽可能在尽可能在雷达导弹连线的右下方
        """
        distance = []
        for agent in self.env.agents:
            d = np.linalg.norm([agent.x - self.x, agent.y - self.y])
            if d < self.Round:
                distance.append(d)
            else:
                distance.append(100000)

    def __reto_motion_2(self) -> int:
        """
        任务姿态对准奖励，Agent的heading对准目标，波束覆盖了目标，获得奖励
        Agent的波束覆盖角度预设为90度，heading两侧各45度
        """

        return 0
    def __reto_motion_3(self):
        """
        agent距离J20和radar连线的距离 奖励
        """

        reward = -5

        return reward



    def __reto_ew_j_1(self):


        return 0


    def __reto_ew_j_2(self):
        """
        连续yazhi时间奖励，T_ew，暂定奖励r=T_ew*1
        """
        return 0

    def __reto_ew_j_3(self):
        """
        掩护目标奖励,包括个体掩护奖励和集群掩护奖励
        """
        total_cover_reward = 0


        return total_cover_reward

    def __remso_in(self):
        """
        平台内电磁兼容奖励，电磁兼容通过，奖励1，冲突，奖励-1
        """
        return 0

    def __remso_out(self):
        """
        平台外电磁兼容奖励，电磁兼容通过，奖励1，冲突，奖励-1
        """
        return 0



    def get_reward(self):
        """
        通过一个"权重词典"，来计算以上所有奖励函数的加权和，enable_dist.values应取自集合{0，1}
        """
        sub_reward = [
            self.__rapo_rwd(),
            self.__space_limit_punishment_x(),
            self.__space_limit_punishment_y(),
            self.__reto_motion_1(),
            self.__reto_motion_2(),
            self.__reto_motion_3(),
            self.__reto_ew_j_1(),
            self.__reto_ew_j_2(),
            self.__reto_ew_j_3(),
            self.__remso_in(),
            self.__remso_out(),
        ]
        if self.reward_weight is not None:
            return np.dot(sub_reward, self.reward_weight)
        else:
            return sum(sub_reward)
