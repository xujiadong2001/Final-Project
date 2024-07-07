
import os
import pickle
'''
def get_frame_numbers(data_dir):
    frame_numbers = {}
    for sample in os.listdir(data_dir):
        sample_dir = os.path.join(data_dir, sample)
        frame_idx = sorted([int(frame.split('.')[0].split('_')[-1]) for frame in os.listdir(sample_dir)])
        # 检查time_series文件夹下的对应的sample_k.pkl文件中frame部分与frame_idx是否一致
        video_name = sample.split('_')[1]
        pkl_dir="collect_331_5D_surface/time_series"
        with open(os.path.join(pkl_dir, f"sample_{video_name}.pkl"), 'rb') as f:
            data = pickle.load(f)
            if data['frame'] != frame_idx:
                print(f"sample: {sample} frame: {frame_idx} data: {data['frame']}")
    return frame_numbers


get_frame_numbers("collect_331_5D_surface/videos")
'''
with open(os.path.join("collect_331_5D_surface/time_series", "sample_96.pkl"), 'rb') as f:
    data = pickle.load(f)
    print(len(data['frame']))
    print(data['frame'])

i=0
for k in data['frame']:
    if k!=i:
        print(k)
    i+=1