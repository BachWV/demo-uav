import numpy as np
import math
import torch


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e ** 2 / 2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def merge_intervals(intervals):
    """
    区间合并函数，将一个列表当中的多个区间，依据是否有重叠部分进行合并
    Example：
    input: test = [
                [1, 3],
                [2, 4],
                [6, 9]
            ]
    output: [[1, 4], [6, 9]]
    """
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:  # 当前区间与结果列表中的最后一个区间不重叠
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def check_suppress_plot(agent_x, agent_y, agent_heading, radar_x, radar_y) -> bool:
    """
    用于在画图过程中，判断当前UAV是否对敌方目标形成了有效yazhi
    """

    # 由于np.arctan()返回的值在区间[-pi/2, pi/2], heading的值在区间[0, 360]，无法直接进行比较,
    # 因此，需要对为负值的角度进行换算，换算成[0, 360]区间内的角度
    # angle为agent向radar的连线与x轴正方向的夹角
    angle_uav2radar = 0
    diff = 0
    if agent_x < radar_x and agent_y > radar_y:
        angle_uav2radar = np.degrees(np.arctan((radar_y - agent_y) / (radar_x - agent_x))) + 360
        diff = np.abs(angle_uav2radar - agent_heading)
        if diff > 180:
            diff = 360 - diff
    elif agent_x < radar_x and agent_y < radar_y:
        angle_uav2radar = np.degrees(np.arctan((radar_y - agent_y) / (radar_x - agent_x)))
        diff = np.abs(angle_uav2radar - agent_heading)
        if diff > 180:
            diff = 360 - diff
    elif agent_x > radar_x and agent_y > radar_y:
        angle_uav2rad = np.degrees(np.arctan((radar_y - agent_y) / (radar_x - agent_x))) + 180
        diff = np.abs(angle_uav2radar - agent_heading)
        if diff > 180:
            diff = 360 - diff
    elif agent_x > radar_x and agent_y < radar_y:
        angle_uav2rad = 180 - np.degrees(np.arctan((radar_y - agent_y) / (radar_x - agent_x)))
        diff = np.abs(angle_uav2radar - agent_heading)
        if diff > 180:
            diff = 360 - diff
    else:
        pass

    # angle_red = np.degrees(np.arctan((radar_y - agent_y) / (radar_x - agent_x)))
    # if angle_red <= 0:
    #     angle_red += 360
    # heading = agent_heading
    # alpha = np.abs(angle_red - heading)

    if diff < 45:
        return True
    else:
        return False
