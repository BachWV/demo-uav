import numpy as np

from utils.util import merge_intervals
from envs.Radar import Radar
from envs.J20 import J20
class Agent(object):
    """
    我方用于掩护的无人机
    """

    def __init__(self, agent_id: int, init_pos: [int, int], env, reward_weight: dict, J_20):
        self.env = env
        self.agent_id = agent_id
        self.x, self.y = init_pos[0], init_pos[1]
        self.vel = 272  # 速度
        self.vel_min = 100
        self.vel_max = 500
        self.acceleration = 20  # 加速度
        self.heading = np.random.randint(-180, 180)   # agent的朝向
        self.angle = 0  # 作用未知
        self.action_heading = 0  # 规划朝向角
        # 决策空间维度，即actor网络输出的动作是一个拟定让UAV飞向的角度，但是实际，UAV无法瞬时转向，
        # 因此agent更新是向action_heading方向进行一step的转向
        self.action_dim = 3
        self.hp = 100

        self.radar = Radar(radar_id=0, x=0, y=0)  # ganrao目标，当前为随意初始化的一个实例， 后续会进行分配
        self.J20 = J_20
        self.obs_J20 = [J_20.x / 1000.0, J_20.y / 1000.0, J_20.vel, J_20.heading]

        self.reward_weight = reward_weight
        self.obs = list()
        self.get_obs()

    def get_obs(self):
        """
        获得当前agent对环境的观测 10维
        """
        scale = 1000.0

        angle_J20 = np.degrees(np.arctan2(self.radar.y - self.J20.y, self.radar.x -  self.J20.x))

        angle_agent = np.degrees(np.arctan2(self.radar.y - self.y, self.radar.x - self.x))

        #radar顶点的夹角
        angle_radar = self.get_vec_angle_cos([self.x - self.radar.x,self.y - self.radar.y],[self.J20.x - self.radar.x,self.J20.y - self.radar.y])



        dx = self.x - self.radar.x
        dy = self.y - self.radar.y

        lenth_agent_radar = np.sqrt( dx*dx/1000000 + dy*dy /1000000)

        lenth_agent_line = np.sqrt(1- angle_radar*angle_radar) * lenth_agent_radar

        self.lenth_agent_line = lenth_agent_line

        angle_J20_agent = np.degrees(np.arctan2(self.y - self.J20.y, self.x -  self.J20.x))


        self.obs = [
            self.vel, self.heading, angle_agent, angle_J20,angle_J20_agent, lenth_agent_line,np.degrees(self.J20.heading), np.degrees(np.arccos(angle_radar))
            # self.x/scale, self.y/scale, self.vel, self.heading,
            # self.radar.x/scale, self.radar.y/scale,
            # self.J20.x/scale, self.J20.y/scale, self.J20.vel, self.J20.heading
        ]


    def update(self, action):
        if self.hp > 0:
            self.move(acc=0, action=action)
        self.get_obs()

    def move(self, acc, action):
        """
        acc: 控制Agent的位移; <0, 减速；>0,加速; =0, 速度不变
        action: 控制Agent的转角，示例[1, 0, 0, ... , 0], 180个元素的one-hot向量，表示action_heading的角度大小，
                映射关系为：[(元素1对应的index) + 1] * 2 = action_heading
        """
        if acc < 0:
            self.vel = max(self.vel_min, self.vel - self.acceleration)  # 加速度20m/(s^2)
        elif acc > 0:
            self.vel = min(self.vel_max, self.vel + self.acceleration)

        # self.action_heading = np.argmax(action) - 15
        self.action_heading = np.argmax(action)
        if self.action_heading == 0:
            self.heading = (self.heading + 15 +180) % 360 -180
        elif self.action_heading == 1:
            self.heading = (self.heading - 15 + 180) %360 -180
        else:
            pass

        self.x += self.vel * np.cos(np.radians(self.heading))
        self.y += self.vel * np.sin(np.radians(self.heading))

    def get_vec_angle_cos(self, v1, v2):
        """
        获取两向量夹角的cos值
        """
        x1, y1 = v1[0], v1[1]
        x2, y2 = v2[0], v2[1]
        if x1 * x1 + y1 * y1 == 0 or x2 * x2 + y2 * y2 == 0:
            return 0
        else:
            cos_t = (x1 * x2 + y1 * y2) / np.sqrt(x1 * x1 + y1 * y1) / np.sqrt(x2 * x2 + y2 * y2)  # v1, v2的夹角t的cos
            return cos_t

    def get_vec_angle(self, v1, v2):
        """
        获取两向量夹角的角度值，输出值在[0, 180]度区间内
        """
        x1, y1 = v1[0], v1[1]
        x2, y2 = v2[0], v2[1]
        if x1 * x1 + y1 * y1 == 0 or x2 * x2 + y2 * y2 == 0:
            return 0
        else:

            # (x1 * x2 + y1 * y2) 可以用np.dot(v1, v2)优化
            cos_t = (x1 * x2 + y1 * y2) / np.sqrt(x1 * x1 + y1 * y1) / np.sqrt(x2 * x2 + y2 * y2)  # v1, v2的夹角t的cos
            angle = np.degrees(np.arccos(cos_t))
            return angle

    def get_angle_diff(self, angle1, angle2):
        """
        获取两个角度的差值，将angle1，angle2转化为角度在一个周期内的单位向量，再调用get_vec_angle求两向量的夹角
        angle1, angle2: 输入为角度制
        """
        v1 = (np.cos(np.radians(angle1)), np.sin(np.radians(angle1)))
        v2 = (np.cos(np.radians(angle2)), np.sin(np.radians(angle2)))
        # -------------------------------
        cos_t = np.dot(v1, v2)
        angle = np.degrees(np.arccos(cos_t))

        return angle

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
        # if np.abs(self.y) > 2 * np.abs(self.radar.y):
        #     return -5
        # elif np.abs(self.y) > 1.2 * np.abs(self.radar.y):
        #     return -2
        # elif np.abs(self.y) > np.abs(self.radar.y):
        #     return 0
        # else:
        #     return 2

    def __space_limit_punishment_x(self):
        return 0
        # """
        # 用来惩罚uav倒飞
        # """
        # if self.x < self.J20.x:
        #     return -100
        # else:
        #     return (self.x-90000)/2000

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
    def check_suppress(self) -> bool:
        """
        用于判断当前UAV是否对敌方目标形成了有效yazhi
        """
        angle = np.degrees(np.arctan2(self.radar.y - self.y, self.radar.x - self.x))
        alpha = self.get_angle_diff(angle, self.heading)
        if alpha < 45:
            return True
        else:
            return False

    def check_cover(self, cover_angle=2.5) -> bool:
        """
        用于判断当前UAV，针对当前的目标radar是否掩护成功了我方目标
        """
        if self.check_suppress():
            v1 = (self.J20.x - self.radar.x, self.J20.y - self.radar.y)
            v2 = (self.x - self.radar.x, self.y - self.radar.y)
            difference = self.get_vec_angle(v1, v2)
            if difference < cover_angle:
                return True
            else:
                return False
        else:
            return False

    def suppress_fan_range(self) -> [float, float]:
        """
        以敌方目标为坐标原点，建立直角坐标系，以[0,360]为范围，返回当前uav对敌方目标的yazhi扇面的角度区间
        yazhi角度，4°
        """
        if self.check_suppress():
            angle = np.degrees(np.arctan2(self.y - self.radar.y, self.x - self.radar.x)) + 180  # [0~360]
            return [angle - 2, angle + 2]
        else:
            return [0, 0]

    def __reto_ew_j_1(self):
        """
        yazhi扇面奖励，连续拼接角度alpha越大，奖励越大
        """
        # 用于存储ganrao当前雷达的所有agent的yazhi扇面角度范围
        all_pan_range = [agent.suppress_fan_range() for agent in self.radar.agents]
        merged_pan_range = merge_intervals(all_pan_range)  # 用于存储合并后的yazhi扇面角度范围
        merged_angle = max([np.abs(x[1] - x[0]) for x in merged_pan_range])  # 返回合并后的范围最大的角度范围

        return merged_angle   # 1~4之间

    def cal_cover_range(self):
        """
        yazhi扇面奖励，连续拼接角度alpha越大，奖励越大
        """
        # 用于存储ganrao当前雷达的所有agent的yazhi扇面角度范围
        all_pan_range = [agent.suppress_fan_range() for agent in self.radar.agents]
        merged_pan_range = merge_intervals(all_pan_range)  # 用于存储合并后的yazhi扇面角度范围
        merged_angle = max([np.abs(x[1] - x[0]) for x in merged_pan_range])  # 返回合并后的范围最大的角度范围

        return int(merged_angle)

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
        swarm_cover_flag = False

        for ag in self.radar.agents:
            if ag.check_cover(5):
                swarm_cover_flag = True
                break
        if swarm_cover_flag:
            total_cover_reward += 10
        else:
            total_cover_reward += -10

        difference = 10
        if self.check_suppress():
            v1 = (self.J20.x - self.radar.x, self.J20.y - self.radar.y)
            v2 = (self.x - self.radar.x, self.y - self.radar.y)
            difference = self.get_vec_angle(v1, v2)

        if difference < 2:
            total_cover_reward += 5
        elif difference < 4:
            total_cover_reward += 2
        elif difference < 6:
            total_cover_reward += 1
        else:
            total_cover_reward += -10

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

    def __communicate_max_range_rwd(self):
        """
        最大通信范围奖励，r = 通信范围 * 1
        """
        return 0

    def __communicate_connectivity(self):
        """
        通信连通度奖励 >1, r = 1, <1, r = -10
        """
        return 0

    def __energy_consumption(self):
        """
        能耗奖励：zhencha能耗，gr能耗，通信能耗
        """
        return 0

    def __endurance_rwd(self):
        """
        续航奖励，续航时间t_l2e, r = t_l2e * 1
        """
        return 0

    def __alive_rwd(self) -> int:
        """
        幸存奖励，每生存一回合，r=1
        """
        return 0

    def __collision_punish(self):
        """
        碰撞惩罚，撞了 惩罚 -10，没撞，但是距离过近 -5，其余 1
        """
        r = 0
        for agent in self.env.agents:
            if agent.agent_id != self.agent_id:
                temp = (self.x - agent.x, self.y - agent.y)
                dis = np.linalg.norm(temp)
                if dis < self.vel * 1:
                    return -10
                elif dis < self.vel * 3:
                    return -5
                else:
                    pass
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
            self.__communicate_max_range_rwd(),
            self.__communicate_connectivity(),
            self.__energy_consumption(),
            self.__endurance_rwd(),
            self.__alive_rwd(),
            self.__collision_punish()
        ]
        if self.reward_weight is not None:
            return np.dot(sub_reward, self.reward_weight)
        else:
            return sum(sub_reward)
