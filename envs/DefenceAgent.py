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

        self.Round = 10_100 # 攻击范围
        self.attack_position = [0, 0] # 攻击目标
        self.attack_index = 0 # 攻击目标的索引
        self.attack_heading = 0 # 攻击目标的角度
        self.avail = 1 # 是否可用

        #攻击区域 attack_area = [x_min, x_max, y_min, y_max]
        self.attack_area = [0, 0, 0, 0]
        self.action_dim = 3

        self.reward_weight = reward_weight
        self.obs = list()
        self.get_obs()

    def get_obs(self):
        """
        获得当前agent对环境的观测 10维
        """
        self.obs = [

        ]


    def update(self, action):
        self.attack(action=action)
        self.get_obs()

    def attack(self, action):
        """
        acc: 控制Agent的位移; <0, 减速；>0,加速; =0, 速度不变
        action: 控制Agent的转角，示例[1, 0, 0, ... , 0], 180个元素的one-hot向量，表示action_heading的角度大小，
                映射关系为：[(元素1对应的index) + 1] * 2 = action_heading
        """
        # 攻击
        distance = [np.linalg.norm([defence.x - self.x, defence.y - self.y]) for defence in self.env.agents]
        # 查找distance中最小值对应的索引
        self.attack_index = distance.index(min(distance))
        # 将对应的agent的hp减少10
        self.env.agents[self.attack_index].hp -= 10





    def is_done(self) -> bool:
        """
        回合结束判定，如果x方向无人机位置超过了雷达位置，就结束当前回合
        """
        # if self.x > self.radar.x:
        #     return True
        # return False
        return self.check_cover(5)

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
        angle_agent = self.obs[2]
        angle_J20 = self.obs[3]
        alpha = angle_agent - angle_J20
        reward = -5
        if alpha > 4:
            reward = 5
        elif alpha > 2:
            reward = 10
        elif alpha > 0:
            reward = 15
        elif alpha > -2:
            reward = 2
        elif alpha > -4:
            reward = 0
        else:
            pass
        return reward

    def __reto_motion_2(self) -> int:
        """
        任务姿态对准奖励，Agent的heading对准目标，波束覆盖了目标，获得奖励
        Agent的波束覆盖角度预设为90度，heading两侧各45度
        """
        angle = np.degrees(np.arctan2(self.radar.y - self.y, self.radar.x - self.x))
        alpha = self.get_angle_diff(angle, self.heading)
        reward = -10
        if alpha < 20:
            reward = 20
        elif alpha < 45:
            reward = 10
        else:
            reward = -5
        if -100e3 < self.x < -50e3:
            reward *= 2
        return reward
    def __reto_motion_3(self):
        """
        agent距离J20和radar连线的距离 奖励
        """
        lenth = self.lenth_agent_line
        reward = -5
        if lenth < 1:
            reward = 10
        elif lenth < 2:
            reward = 5
        elif lenth < 6:
            reward = 1
        elif lenth < 8:
            reward = 0
        else:
            reward = -5
        if -100e3 < self.x < -50e3:
            reward *= 2
        return reward



    def __reto_ew_j_1(self):
        """
        yazhi扇面奖励，连续拼接角度alpha越大，奖励越大
        """
        # 用于存储ganrao当前雷达的所有agent的yazhi扇面角度范围
        all_pan_range = [agent.suppress_fan_range() for agent in self.radar.agents]
        merged_pan_range = merge_intervals(all_pan_range)  # 用于存储合并后的yazhi扇面角度范围
        merged_angle = max([np.abs(x[1] - x[0]) for x in merged_pan_range])  # 返回合并后的范围最大的角度范围

        return merged_angle   # 1~4之间


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
