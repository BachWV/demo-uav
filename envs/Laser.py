class Laser(object):
    def __init__(self, x, y, r):
        self.x = 0
        self.y = 0
        self.r = r
        self.state = 0 # 0: 未激光 1: 激光
    def step(self, action):
        if action == 1:
            self.state = 1
        else:
            self.state = 0
        self.laser.update()
        return super().step(action)


class LaserControl(object):
    def __init__(self, x, y, r):
        self.laser = Laser(x, y, r)
    def step(self, action):
        self.laser.step(action)
        return self.laser.x, self.laser.y, self.laser.r
    def reset(self):
        self.laser.x = 0
        self.laser.y = 0
        self.laser.r = 0
        return self.laser.x, self.laser.y, self.laser.r
    def render(self):
        pass
    def close(self):
        pass
    def seed(self):
        pass
    def close(self):
        pass