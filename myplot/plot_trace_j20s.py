"""
画出trace_file中各个物体的轨迹以及角度关系
"""
import pickle
import socket
import sys

import numpy as np
import matplotlib.lines
import matplotlib.pyplot as plt

from matplotlib import animation as ani
from matplotlib import patches
from utils.util import check_suppress_plot

trace_file = '../trace/weight01/render/run4.json'


class Agent(object):

    def __init__(self, axes, dic):
        self.ax = axes
        self.X = dic['x']
        self.Y = dic['y']
        self.Heading = dic['heading']
        self.R = dic['reward']
        self.hp = dic['hp']
        self.plt_sca = self.ax.scatter(0, 0)
        self.plt_trj, = self.ax.plot(0, 0)
        self.wedge = patches.Wedge(center=(-1e7, -1e7), r=5e3, theta1=0, theta2=0, ec='none', color='r', alpha=0.3)
        self.ax.add_patch(self.wedge)
        # self.plt_txt = self.ax.text(x,y,'%d'%(r),bbox=dict(boxstyle='round',fc='w'),fontsize=8)

    def update(self, itr):
        hp = self.hp[itr] // 10
        if hp > 0:
            self.plt_sca.set_offsets([self.X[itr], self.Y[itr]])

            self.plt_trj.set_data([self.X[itr - hp:itr], self.Y[itr - hp:itr]])
            # 更新扫描
            self.wedge.set_center((self.X[itr], self.Y[itr]))
            self.scan_center = self.Heading[itr]
            self.wedge.set_theta1(self.scan_center - 45)
            self.wedge.set_theta2(self.scan_center + 45)
        else:
            self.plt_sca.set_offsets([0, 0])
            self.plt_trj.set_data([0, 0])
            self.wedge.set_center((-1e7, -1e7))

        '''
        self.plt_txt[idx].set_x(x)
        self.plt_txt[idx].set_y(y)
        self.plt_txt[idx].set_text('%d'%(r))
        '''
class DefenceAgent(object):
    """
    我方激光
    """

    def __init__(self, axes, dic):
        self.ax = axes
        self.X = dic['x']
        self.Y = dic['y']
        self.attack_x = dic['attack_x']
        self.attack_y = dic['attack_y']
        self.plt_sca = self.ax.scatter(self.X[0], self.Y[0], color='r') # plt_scatter 是什么
        self.plt_trj, = self.ax.plot(self.X[0], self.Y[0], color='r')



    def update(self, itr):
        pass


class J20(object):

    def __init__(self, ax, dic):
        self.ax = ax
        self.X = dic['x']
        self.Y = dic['y']
        self.plt_sca = self.ax.scatter(0, 0, color='r')
        self.plt_trj, = self.ax.plot(0, 0, color = 'y')

    def update(self, itr):
        self.plt_sca.set_offsets([self.X[itr], self.Y[itr]])
        self.plt_trj.set_data([self.X[0:itr], self.Y[0:itr]])




class Radar(object):
    def __init__(self, ax, dic):
        self.ax = ax
        self.X = dic['x']
        self.Y = dic['y']
        self.plt_sca = self.ax.scatter(0, 0, color='b')
        # self.scanCenter = 180
        # self.wedge = patches.Wedge(center=(-1e7, -1e7), r=100e3, theta1=0, theta2=0, ec='none', color='b', alpha=0.3)
        # self.ax.add_patch(self.wedge)

    def update(self, itr):
        self.plt_sca.set_offsets([self.X[itr], self.Y[itr]])
        # 更新扫描
        # self.wedge.set_center((self.X[itr], self.Y[itr]))
        # # self.scanCenter += 5
        # self.wedge.set_theta1(self.scanCenter - 5)
        # self.wedge.set_theta2(self.scanCenter + 5)


class Line(object):
    """
    用来画两个Radar和J20之间的连线
    """

    def __init__(self, axes, dic_defence_agents):
        self.ax = axes
        self.dic_defence_agents = dic_defence_agents
        self.lines = []
        #遍历dic_defence_agents中的每一个defence_agent


        for defence_agent in self.dic_defence_agents.values():
            self.line_plane2radar_1 = matplotlib.lines.Line2D(xdata=[defence_agent['x'][0], defence_agent['attack_x'][0]],
                                                                ydata=[defence_agent['y'][0], defence_agent['attack_y'][0]])
            self.lines.append(self.line_plane2radar_1)
            self.ax.add_line(self.line_plane2radar_1)

    def update(self, itr):
        for line,defence_agent in zip(self.lines,self.dic_defence_agents.values()):
            line.set(xdata=[defence_agent['x'][itr], defence_agent['attack_x'][itr]],
                     ydata=[defence_agent['y'][itr], defence_agent['attack_y'][itr]])




class SuppressFan(object):
    """
    用于绘制UAV对Radar的yazhi扇面
    """

    def __init__(self, axes, dic_agents, dic_radar_1):
        self.ax = axes
        self.dic_agents = dic_agents
        self.dic_radar_1 = dic_radar_1
        self.wedges_1 = [
            patches.Wedge(
                center=(dic_radar_1["x"][0], dic_radar_1["y"][0]),
                r=10 * np.linalg.norm([dic_agents[str(i)]["x"][0] - dic_radar_1["x"][0],
                                       dic_agents[str(i)]["y"][0] - dic_radar_1["y"][0]]),
                theta1=180 + np.degrees(
                    np.arctan((dic_agents[str(i)]["y"][0] - dic_radar_1["y"][0]) /
                              (dic_agents[str(i)]["x"][0] - dic_radar_1["x"][0]))
                ) - 2,
                theta2=180 + np.degrees(
                    np.arctan((dic_agents[str(i)]["y"][0] - dic_radar_1["y"][0]) /
                              (dic_agents[str(i)]["x"][0] - dic_radar_1["x"][0]))
                ) + 2,
                ec='none', color='b',
                alpha=0.3 if check_suppress_plot(dic_agents[str(i)]["x"][0], dic_agents[str(i)]["y"][0],
                                                 dic_agents[str(i)]["heading"][0],
                                                 dic_radar_1["x"][0], dic_radar_1["y"][0]) else 0
            ) for i in range(len(dic_agents))
        ]
        for wedge in self.wedges_1:
            self.ax.add_patch(wedge)


    def update(self, itr):
        for i in range(len(self.dic_agents)):
            # -------------------以下是关于radar1yazhi扇面的绘制-------------------- #
            if self.dic_agents[str(i)]["x"][itr] < self.dic_radar_1["x"][itr]:
                self.wedges_1[i].set_center((self.dic_radar_1["x"][itr], self.dic_radar_1["y"][itr]))
                self.wedges_1[i].set_theta1(
                    180 + np.degrees(
                        np.arctan((self.dic_agents[str(i)]["y"][itr] - self.dic_radar_1["y"][itr]) /
                                  (self.dic_agents[str(i)]["x"][itr] - self.dic_radar_1["x"][itr]))
                    ) - 3
                )
                self.wedges_1[i].set_theta2(
                    180 + np.degrees(
                        np.arctan((self.dic_agents[str(i)]["y"][itr] - self.dic_radar_1["y"][itr]) /
                                  (self.dic_agents[str(i)]["x"][itr] - self.dic_radar_1["x"][itr]))
                    ) + 3
                )
                self.wedges_1[i].set_alpha(
                    0.3 if check_suppress_plot(self.dic_agents[str(i)]["x"][itr], self.dic_agents[str(i)]["y"][itr],
                                               self.dic_agents[str(i)]["heading"][itr],
                                               self.dic_radar_1["x"][itr], self.dic_radar_1["y"][itr]) else 0)



class PlotOffLine(object):
    def __init__(self, trace_file):
        self.fig = plt.figure(figsize=(20,8))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlim([-100e3, 10e3])
        self.ax.set_ylim([-50e3, 50e3])
        self.ax.grid(True, linestyle='--')
        self.load_data(trace_file)
        self.dic_radar_1 = {}
        self.dic_radar_plane = {}
        self.line1 = Line(
            axes=self.ax,
            dic_defence_agents=self.dic['defence_agents']
        )



        A = ani.FuncAnimation(fig=self.fig, func=self.update, frames=int(300))
        # A.save("newmap35e3_18.gif",writer='pillow',dpi=100)
        plt.show()

    def load_data(self, trace_file):
        import json
        self.entity = []
        with open(trace_file) as fo:
            self.dic = json.load(fo)
            # if 'J20' in self.dic:
            #     plane0 = J20(ax=self.ax, dic=self.dic['J20']['0'])
            #     self.entity.append(plane0)


            if 'radars' in self.dic:
                for radar_id in self.dic['radars']:
                    radar = Radar(ax=self.ax, dic=self.dic['radars'][radar_id])
                    self.entity.append(radar)
            if 'agents' in self.dic:
                for ag_id in self.dic['agents']:
                    ag = Agent(axes=self.ax, dic=self.dic['agents'][ag_id])
                    self.entity.append(ag)
            if 'defence_agents' in self.dic:
                for ag_id in self.dic['defence_agents']:
                    ag = DefenceAgent(axes=self.ax, dic=self.dic['defence_agents'][ag_id])
                    self.entity.append(ag)

    def update(self, itr):
        for entity in self.entity:
            # if isinstance(entity, J20):
            #     print(len(entity.X))
            #     # temp["J20"].append([entity.X, entity.Y])
            # elif isinstance(entity, Radar):
            #     # temp["Radar"].append([entity.X, entity.Y])
            #     # print(len(entity.X))
            entity.update(itr)
        self.line1.update(itr)

        # self.ax.plot(temp["J20"][0], temp["Radar"][0])
        # self.ax.plot(temp["J20"][0], temp["Radar"][1])
        # print(temp)


class PlotOnLine(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([0, 200e3])
        self.ax.set_ylim([-100e3, 100e3])
        self.ax.grid(True, linestyle='--')
        self.plt = {}
        self.__udp_init__()
        A = ani.FuncAnimation(fig=self.fig, func=self.update, frames=range(1, 200))
        plt.show()

    def __udp_init__(self):
        recvPort = ('127.0.0.1', 4000)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(recvPort)

    def update(self, itr):
        msgRecv, _ = self.udp.recvfrom(65536)
        dic = pickle.loads(msgRecv)
        for idx in dic:
            x, y, theta = dic[idx][0], dic[idx][1], dic[idx][2]
            if idx not in self.plt:
                self.plt[idx] = self.ax.scatter(x, y)
            self.plt[idx].set_offsets([x, y])
        print(dic)


if __name__ == "__main__":
    # 科目六 3
    # trace_file = '../trace/weight01/render/run21.json'
    s = PlotOffLine(trace_file)
