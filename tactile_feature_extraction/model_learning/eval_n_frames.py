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
from tactile_feature_extraction.pytorch_models.supervised.image_generator import PhotoDataset, PhotoDataset_ConvLstm, PhotoDataset_ConvLstm_2, PhotoDataset_Seq2Seq

from tactile_feature_extraction.utils.utils_learning import FTPoseEncoder
from tactile_feature_extraction.utils.utils_learning import seed_everything, make_dir
from tactile_feature_extraction.utils.utils_learning import get_ft_pose_limits
from tactile_feature_extraction.utils.utils_learning import make_save_dir_str

from tactile_feature_extraction import BASE_DATA_PATH
from tactile_feature_extraction import BASE_MODEL_PATH

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def rename_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"File renamed from {old_name} to {new_name}")
    except FileNotFoundError:
        print(f"The file {old_name} does not exist")
    except Exception as e:
        print(f"Error occurred while renaming file: {e}")

def launch():
    args = parse_args()
    async_data = args.async_data
    tasks = args.tasks
    sensors = args.sensors
    models = args.models
    device = args.device

    for task in tasks:
        results = {}
        for model_type in models:
            MAE = []
            Acc = []
            MSE = []
            R_square = []
            out_dim, label_names = setup_task(task)
            save_dir_str = make_save_dir_str(async_data, task, sensors)
            save_dir = os.path.join(BASE_MODEL_PATH, save_dir_str, model_type)
            make_dir(save_dir)

            model_params = setup_model(model_type, save_dir)
            learning_params, image_processing_params, frame_processing_params, augmentation_params = setup_learning(save_dir)

            n_frames_list = [3, 5, 7, 9, 11, 13]
            PhotoDataset_ConvLstm_list = ['conv_lstm', 'conv_transformer', 'conv_gru', 'CNN3D', 'conv_gru_attention', 'conv3d_gru', 'r_convlstm', 'conv_TCN', 'conv_lstm_attention', 'TimeAttention']
            PhotoDataset_Seq2Seq_list = ['seq2seq_gru', 'seq2seq_gru_attention', 'seq2seq_transformer']

            if model_type in PhotoDataset_ConvLstm_list:
                DataGenerator = PhotoDataset_ConvLstm
            elif model_type in PhotoDataset_Seq2Seq_list:
                DataGenerator = PhotoDataset_Seq2Seq
            else:
                DataGenerator = PhotoDataset

            in_dim = image_processing_params['dims']
            in_channels = 1
            combined_dirs = list(itertools.product(["linshear_surface_3d"], sensors))
            combined_paths = [os.path.join(*i) for i in combined_dirs]

            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, data_path, "val")
                for data_path in combined_paths
            ]
            photos_dir = 'collect_331_5D_surface/videos'
            labels_dir = 'collect_331_5D_surface/time_series'

            video_list = [f for f in os.listdir(photos_dir)]
            val_video_list = video_list[3 * int(len(video_list) / 5):(3 + 1) * int(len(video_list) / 5)]
            train_video_list = [i for i in video_list if i not in val_video_list]

            np.random.shuffle(video_list)
            ft_pose_limits = get_ft_pose_limits(val_data_dirs, save_dir)

            for n_frames in n_frames_list:
                # 读取已保存的模型
                model = create_model(
                    in_dim=in_dim,
                    in_channels=in_channels,
                    out_dim=out_dim,
                    model_params=model_params,
                    device=device,
                )

                val_generator = DataGenerator(
                    photos_dir,
                    labels_dir,
                    n_frames,
                    video_list=val_video_list,
                    padding=True,
                    **image_processing_params
                )

                label_encoder = FTPoseEncoder(label_names, ft_pose_limits, device)

                # 加载模型权重
                best_model_path = os.path.join(save_dir, f'best_model.pth_{n_frames}')
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path))
                    model.eval()
                    print(f'Evaluating model for {n_frames} frames...')

                    eval_result = evaluate_model(
                        task,
                        model_type,
                        model,
                        label_encoder,
                        val_generator,
                        learning_params,
                        save_dir,
                        None,
                        device=device,
                        return_result=True
                    )

                    MAE.append(eval_result[0])
                    MSE.append(eval_result[1])
                    Acc.append(eval_result[2])
                    R_square.append(eval_result[3])

                else:
                    print(f"Model for {n_frames} frames not found, skipping evaluation.")

            # 保存结果
            with open(os.path.join(save_dir, 'model_result_evaluation.txt'), 'w') as f:
                f.write(f'{n_frames}:\n')
                f.write('MAE:' + str(MAE) + '\n')
                f.write('MAE_mean:' + str(np.mean(MAE)) + '\n')
                f.write('MSE:' + str(MSE) + '\n')
                f.write('MSE_mean:' + str(np.mean(MSE)) + '\n')
                f.write('Acc:' + str(Acc) + '\n')
                f.write('Acc_mean:' + str(np.mean(Acc)) + '\n')
                f.write('R_square:' + str(R_square) + '\n')
                f.write('R_square_mean:' + str(np.mean(R_square)) + '\n')

if __name__ == "__main__":
    launch()
