import os
import shutil
import pandas as pd

# 假设 BASE_DATA_PATH 和 TIP_ID 已经在模块 Main_Code 中定义
from tactile_feature_extraction import BASE_DATA_PATH
from tactile_feature_extraction import TIP_ID


def col_rename(path):
    df = pd.read_csv(f'{path}/targets.csv')
    names = []

    for i, _ in enumerate(df['data_name']):
        names.append(f'feature_{i}.npy')

    df.insert(2, 'sensor_features', names, True)
    df.to_csv(f'{path}/targets.csv', index=False)


def move_and_rename(targetPath, featurePath, newPath, processed):
    df = pd.read_csv(f'{targetPath}/targets.csv')
    data_names = df['data_name'].tolist()
    file_names = os.listdir(featurePath)

    i = 0
    for dataname in data_names:
        dataname_num = int(''.join(filter(str.isdigit, dataname)))  # 提取数字部分
        for filename in file_names:
            filename_num = int(''.join(filter(str.isdigit, filename)))  # 提取数字部分
            if processed:
                if dataname_num == filename_num:
                    shutil.copy(f'{featurePath}/{filename}', f'{newPath}/feature_{i}.npy')
                    i += 1
            else:
                if dataname_num == filename_num:
                    shutil.move(f'{featurePath}/{filename}', f'{newPath}/feature_{i}.npy')
                    i += 1


# Set root path
dataPath_0 = BASE_DATA_PATH
dataPath = os.path.join(BASE_DATA_PATH, 'linshear_surface_3d', TIP_ID)

# Define train & validation paths
trainDir = f'{dataPath}/train'
valDir = f'{dataPath}/val'

# Define old feature paths (to be moved)
featurePath = f'{dataPath_0}/features'

# Define and make new feature paths for:
trainFeatureDir = os.path.join(trainDir, 'features')
valFeatureDir = os.path.join(valDir, 'features')

print('Renaming columns for training and validation datasets')
col_rename(trainDir)
col_rename(valDir)

try:
    os.mkdir(trainFeatureDir)
    os.mkdir(valFeatureDir)
except FileExistsError:
    pass

print('Moving training features')
move_and_rename(trainDir, featurePath, trainFeatureDir, processed=False)
print('Moving validation features')
move_and_rename(valDir, featurePath, valFeatureDir, processed=False)
