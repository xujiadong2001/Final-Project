import os
import itertools
import torch
import numpy as np

from tactile_feature_extraction.model_learning.setup_learning import parse_args
from tactile_feature_extraction.model_learning.setup_learning import setup_model
from tactile_feature_extraction.model_learning.setup_learning import setup_learning
from tactile_feature_extraction.model_learning.setup_learning import setup_task
from tactile_feature_extraction.model_learning.evaluate_model import evaluate_model

from tactile_feature_extraction.pytorch_models.supervised.models import create_model
from tactile_feature_extraction.pytorch_models.supervised.train_model_w_metrics import train_model_w_metrics
from tactile_feature_extraction.pytorch_models.supervised.image_generator import PhotoDataset_combinexy, PhotoDataset_ConvLstm_combinexy, PhotoDataset_ConvLstm_2, PhotoDataset_Seq2Seq_combinexy


from tactile_feature_extraction.utils.utils_plots import ErrorPlotter
from tactile_feature_extraction.utils.utils_learning import FTPoseEncoder
from tactile_feature_extraction.utils.utils_learning import seed_everything, make_dir
from tactile_feature_extraction.utils.utils_learning import get_ft_pose_limits
from tactile_feature_extraction.utils.utils_learning import csv_row_to_label
from tactile_feature_extraction.utils.utils_learning import make_save_dir_str

from tactile_feature_extraction import BASE_DATA_PATH
from tactile_feature_extraction import BASE_MODEL_PATH
from tactile_feature_extraction import SAVED_MODEL_PATH

photos_dir = 'collect_331_5D_surface/videos'
labels_dir = 'collect_331_5D_surface/time_series'
n_frames = 10
train_video_list = [f for f in os.listdir(photos_dir)]
train_generator = PhotoDataset_combinexy(
                    photos_dir,
                    labels_dir,
                    n_frames,
                    video_list=train_video_list,
                    padding=True, # 舍弃不足n帧的数据
                )
# 找到“Fxy”的最大值和最小值
min=100
max=-100
for i in range(len(train_generator)):
    labels = train_generator[i]['labels']
    fxy=labels['Fxy']
    if fxy>max:
        max=fxy
    if fxy<min:
        min=fxy
print(min,max)