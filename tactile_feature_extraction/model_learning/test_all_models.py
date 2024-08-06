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
from tactile_feature_extraction.pytorch_models.supervised.image_generator import PhotoDataset, PhotoDataset_ConvLstm, PhotoDataset_ConvLstm_2, PhotoDataset_Seq2Seq


from tactile_feature_extraction.utils.utils_plots import ErrorPlotter
from tactile_feature_extraction.utils.utils_learning import FTPoseEncoder
from tactile_feature_extraction.utils.utils_learning import seed_everything, make_dir
from tactile_feature_extraction.utils.utils_learning import get_ft_pose_limits
from tactile_feature_extraction.utils.utils_learning import csv_row_to_label
from tactile_feature_extraction.utils.utils_learning import make_save_dir_str

from tactile_feature_extraction import BASE_DATA_PATH
from tactile_feature_extraction import BASE_MODEL_PATH
from tactile_feature_extraction import SAVED_MODEL_PATH

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def launch():

    args = parse_args()
    async_data = args.async_data
    tasks = args.tasks
    sensors = args.sensors
    models = args.models
    device = args.device

    for task in tasks:
        results= {} # {'CNN':[],'LSTM':[],...}
        for model_type in models:
            MAE=[]
            Acc=[]
            MSE=[]
            R_square=[]
            # 考虑：鲁棒性（不同分组的效果波动）
            # task specific parameters
            out_dim, label_names = setup_task(task)
            save_dir_str = make_save_dir_str(async_data, task, sensors)
            # setup save dir
            save_dir = os.path.join(BASE_MODEL_PATH, save_dir_str, model_type)
            make_dir(save_dir)

            # setup parameters
            model_params = setup_model(model_type, save_dir)
            learning_params, image_processing_params, frame_processing_params, augmentation_params = setup_learning(save_dir)

            n_frames = 5  # 前n帧

            PhotoDataset_ConvLstm_list = ['conv_lstm', 'conv_transformer', 'conv_gru', 'CNN3D','conv_gru_attention','conv3d_gru','r_convlstm','conv_TCN','conv_lstm_attention','TimeAttention']
            PhotoDataset_Seq2Seq_list = ['seq2seq_gru', 'seq2seq_gru_attention', 'seq2seq_transformer']

            if model_type in PhotoDataset_ConvLstm_list:
                if n_frames <= 20:
                    DataGenerator = PhotoDataset_ConvLstm
                else:
                    DataGenerator = PhotoDataset_ConvLstm_2
            elif model_type in PhotoDataset_Seq2Seq_list:
                DataGenerator = PhotoDataset_Seq2Seq
            else:
                DataGenerator = PhotoDataset

            train_processing_params = {**image_processing_params, **augmentation_params}
            val_processing_params = image_processing_params
            in_dim = image_processing_params['dims']
            in_channels = 1

            # create the model
            seed_everything(learning_params['seed'])

            error_plotter = None
            combined_dirs = list(itertools.product(["linshear_surface_3d"], sensors))
            combined_paths = [os.path.join(*i) for i in combined_dirs]

            train_data_dirs = [
                os.path.join(BASE_DATA_PATH, data_path, "train")
                for data_path in combined_paths
            ]
            ft_pose_limits = get_ft_pose_limits(train_data_dirs, save_dir)

            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, data_path, "val")
                for data_path in combined_paths
            ]

            photos_dir = 'collect_331_5D_surface/videos'
            labels_dir = 'collect_331_5D_surface/time_series'

            video_list = [f for f in os.listdir(photos_dir)]
            # 打乱数据


            np.random.shuffle(video_list)
            # train_video_list = video_list[:int(len(video_list)*0.8)]
            # val_video_list = video_list[int(len(video_list)*0.8):]
            # K-FOLDs
            for index in range(5):
                model = create_model(
                    in_dim=in_dim,
                    in_channels=in_channels,
                    out_dim=out_dim,
                    model_params=model_params,
                    device=device,
                    # cnn_model_dir="collect_331_5D_surface/model/non_async/"+task+"/331/simple_cnn/best_model.pth"
                )

                val_video_list = video_list[index*int(len(video_list)/5):(index+1)*int(len(video_list)/5)]
                train_video_list = [i for i in video_list if i not in val_video_list]

                # set generators and loaders
                train_generator = DataGenerator(
                    photos_dir,
                    labels_dir,
                    n_frames,
                    video_list=train_video_list,
                    padding=True, # 舍弃不足n帧的数据
                    **train_processing_params
                )
                val_generator = DataGenerator(
                    photos_dir,
                    labels_dir,
                    n_frames,
                    video_list=val_video_list,
                    padding=True, # 舍弃不足n帧的数据
                    **val_processing_params
                )
                # 划分训练集和验证集
                # train_size = int(0.8 * len(train_generator))
                # val_size = len(train_generator) - train_size
                # train_generator, val_generator = torch.utils.data.random_split(train_generator, [train_size, val_size])
                # create the encoder/decoder for labels
                label_encoder = FTPoseEncoder(label_names, ft_pose_limits, device)
                '''
                # create instance for plotting errors
                error_plotter = ErrorPlotter(
                    target_label_names=label_names,
                    save_dir=save_dir,
                    name='error_plot.png',
                    plot_during_training=True
                )
                '''
                # Train model
                train_model_w_metrics(
                    model_type,
                    model,
                    label_encoder,
                    train_generator,
                    val_generator,
                    learning_params,
                    save_dir,
                    error_plotter=error_plotter,
                    calculate_train_metrics=False,
                    device=device,
                )
                print('evaluating the final model')
                # perform a final evaluation using the best modely
                # 创建result.txt
                with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
                    f.write('Final evaluation\n')

                model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
                model.eval()
                print('evaluating the best model')
                eval_result=evaluate_model(
                    task,
                    model_type,
                    model,
                    label_encoder,
                    val_generator,
                    learning_params,
                    save_dir,
                    error_plotter,
                    device=device,
                    return_result= True
                )
                MAE.append(eval_result[0])
                MSE.append(eval_result[1])
                Acc.append(eval_result[2])
                R_square.append(eval_result[3])
                model.train()
            with open(os.path.join(save_dir, 'model_result.txt'), 'w') as f:
                f.write(model_type+':\n')
                f.write('MAE:'+str(MAE)+'\n')
                f.write('MAE_mean:'+str(np.mean(MAE))+'\n')
                f.write('MSE:'+str(MSE)+'\n')
                f.write('MSE_mean:'+str(np.mean(MSE))+'\n')
                f.write('Acc:'+str(Acc)+'\n')
                f.write('Acc_mean:'+str(np.mean(Acc))+'\n')
                f.write('R_square:'+str(R_square)+'\n')
                f.write('R_square_mean:'+str(np.mean(R_square))+'\n')

            # 先转换为数组再求平均，也要保存每折的结果！
        # results保存为文件

if __name__ == "__main__":
    launch()
