import os
import json
import matplotlib.pyplot as plt
import numpy as np


def total_cover_rate(cover_rate):
    # 创建数据
    x = [i for i in range(1, 101)]

    start_end = [
        [0,250],
        [0,150],
        [150,250],
        [143, 213]
    ]
    ans = []

    for tt in start_end:
        start = tt[0]
        end = tt[1]
        avg_total_cover_rate_0_200 = []
        for episode in cover_rate:
            avg_total_cover_rate_0_200.append(np.mean(episode[start:end]))

        print(f'{start}-{end}step: ', np.mean(avg_total_cover_rate_0_200))
        ans.append(avg_total_cover_rate_0_200)


    is_cover = []
    for episode in cover_rate:
        if sum(episode[143:213]) == 70:
            is_cover.append(1)
        else:
            is_cover.append(0)
    print('143-213任务完成率', np.mean(is_cover))

    # 绘制曲线和拟合直线
    plt.plot(x, ans[0], marker='v', markersize=8, label='0-250')
    plt.plot(x, ans[1], marker='X', markersize=8, label='0-150')
    plt.plot(x, ans[2], marker='P', markersize=8, label='150-250')
    plt.plot(x, ans[3], marker='D', markersize=8, label='143-213')
    # 添加图例和标签
    plt.legend()
    plt.xlabel('Experiment Round')
    plt.ylabel('Task Completion Rate')

   # plt.ylim([0, 1])
    # 显示图形
    plt.show()



data_path_root = r'trace/weight01/render'

datas_files = os.listdir(data_path_root)

cover_rate = []

radar_cover_range = []


for filename in datas_files:
    trace_file = data_path_root + '/' + filename
    with open(trace_file) as fo:
        dic = json.load(fo)
        # if dic['J20']['cover_rate'].__len__() != 200 or dic['J20']['total_cover_range'].__len__() != 200:
        #     print(filename, 'length is not 200')
        #     fo.close()
        #    continue
        cover_rate.append(dic['J20']['cover_rate'])
        radar_cover_range.append(dic['J20']['total_cover_range'])

fo.close()

print('掩护覆盖率')
total_cover_rate(cover_rate)


print('雷达掩护扇面角')
total_cover_rate(radar_cover_range)


