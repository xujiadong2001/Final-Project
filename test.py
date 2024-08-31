# 数据集：collect_331_5D_surface\videos
# video_list = [f for f in os.listdir(photos_dir)]
# 读取每个视频的帧数（文件数）

import os

photos_dir = 'collect_331_5D_surface/videos'
video_list = [f for f in os.listdir(photos_dir)]
video_list.sort()
frames_list = []
for video in video_list:
    frames = len(os.listdir(os.path.join(photos_dir, video)))
    frames_list.append(frames)
# 绘制直方图
import matplotlib.pyplot as plt
import numpy as np
plt.hist(frames_list, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
plt.xlabel('Frames')
plt.ylabel('Frequency')
plt.title('Distribution of frames')
plt.show()

# 输出最大和最小帧数
print(f'Max frames: {max(frames_list)}')
print(f'Min frames: {min(frames_list)}')
