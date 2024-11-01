import numpy as np

class Radar(object):
    """
    敌方用来tance的雷达
    """

    def __init__(self, radar_id, x, y):
        self.agents = []
        self.radar_id = radar_id
        self.init_x = x
        self.init_y = y
        self.x = x
        self.y = y
        self.itr = 0

    def update(self):
        pass

