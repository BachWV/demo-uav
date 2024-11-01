import math
import numpy as np


def norm(x, y):
    return math.sqrt(x * x + y * y)


def calDistance(x1, y1, x2, y2):
    return norm(x1 - x2, y1 - y2)  # 平面坐标系

# def get_angle(x, y):
#     '''
#     return the angle, [0,360]
#     '''
#     theta = 0
#     if x == 0 and y > 0:
#         theta = 90
#     elif x == 0 and y < 0:
#         theta = -90
#     elif x == 0 and y == 0:
#         theta = 0
#     else:
#         theta = 180 * math.atan(y / x) / math.pi
#         if x < 0 and y >= 0:
#             theta += 180
#         if x < 0 and y <= 0:
#             theta -= 180
#     return theta % 360


def get_angle(dx, dy):
    angle = np.degrees(np.arctan2(dx, dy))
    angle = (angle + 360) % 360
    return angle


def cal_distance(A, B, C):
    BA = A - B
    BC = C - B
    BA_length = np.linalg.norm(BA)
    projection_length = np.dot(BA, BC) / BA_length
    projection_vector = (projection_length / BA_length) * BA
    perpendicular_vector = BC - projection_vector
    distance = np.linalg.norm(perpendicular_vector)
    return distance

A = np.array([0,2])
B = np.array([2,0])
C = np.array([2,2])
ans = cal_distance(A,B,C)
print(ans)
# 判断两个无人机是否在通信范围内，如果UAV_agent2为死亡状态，那么直接返回false
def judge_communication_distance(UAV_agent1, UAV_agent2):
    # if UAV_agent2.health == False:
    #     return False
    # else:
    #     dis = Fun.calDistance(UAV_agent1.x, UAV_agent1.y, UAV_agent2.x, UAV_agent2.y)
    #     # 通信距离为30KM
    #     if dis >= 30e3:
    #         return False
    return True
