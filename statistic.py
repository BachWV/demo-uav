import os
import json
import matplotlib.pyplot as plt
import numpy as np


def total_cover_rate(rate):
    # 创建数据
    x = [i for i in range(len(rate))]

    ans = []

    for file in rate:
        alive = 0
        sum = 0
        for agent in file.values():
            hp = agent['hp'][-1]
            if hp > 1:
                alive += 1
            sum += 1
        a = 1 - alive / sum
        print(a)
        ans.append(a)


    print('击毁率', np.mean(ans))

    # 绘制曲线和拟合直线
    plt.plot(x, ans, marker='v', markersize=8, label='0-250')
    # 添加图例和标签
    plt.legend()
    plt.xlabel('Experiment Round')
    plt.ylabel('Task Completion Rate')

   # plt.ylim([0, 1])
    # 显示图形
    plt.show()



data_path_root = r'trace/weight01/render68300'

datas_files = os.listdir(data_path_root)

rate = []

for filename in datas_files:
    trace_file = data_path_root + '/' + filename
    with open(trace_file) as fo:
        dic = json.load(fo)
        rate.append(dic['agents'])

fo.close()


total_cover_rate(rate)

