import numpy as np


class J20(object):
    """
    我方被掩护的目标
    """

    def __init__(self, j20_id, x, y, target):
        self.j20_id = j20_id
        self.x = x
        self.y = y
        self.vel = 700  # 速度，单位是m/s
        self.target = target  # 飞行的目标点坐标
        # 弧度制
        self.heading = np.arctan2((target[1] - self.y), (target[0] - self.x))  # 与x轴正方向的夹角的弧度[-pi, +pi]
        self.cover_rate = 0
        self.total_cover_range = 0
        self.itr = 0


    def update(self):
        """
        更新J20的x，y坐标
        """
        self.itr += 1
        # print("itr = ",self.itr)
        t = self.heading
        self.x += self.vel * np.cos(t)  # 使用速度的x方向分量更新x坐标 np.cos()的输入应为弧度制
        self.y += self.vel * np.sin(t)  # 使用速度的y方向分量更新y坐标

    def set_cover_rate(self, cover_rate):
        """
        由于J20类内无法计算当前状态的cover_rate，因此，计划在Swarm类中计算，通过此函数进行更新
        """
        self.cover_rate = cover_rate


    def set_total_cover_range(self, cover_rate):
        """
        由于J20类内无法计算当前状态的cover_rate，因此，计划在Swarm类中计算，通过此函数进行更新
        """
        self.total_cover_range = cover_rate