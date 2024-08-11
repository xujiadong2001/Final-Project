import argparse
import json
import os
import numpy as np
import pandas as pd
import random
import shutil
import torch

from tactile_feature_extraction.utils.utils_params import POS_TOL
from tactile_feature_extraction.utils.utils_params import ROT_TOL
from tactile_feature_extraction.utils.utils_params import FORCE_TOL
from tactile_feature_extraction.utils.utils_params import TORQUE_TOL
from tactile_feature_extraction.utils.utils_params import POS_LABEL_NAMES
from tactile_feature_extraction.utils.utils_params import ROT_LABEL_NAMES
from tactile_feature_extraction.utils.utils_params import FORCE_LABEL_NAMES
from tactile_feature_extraction.utils.utils_params import TORQUE_LABEL_NAMES
from tactile_feature_extraction.utils.utils_params import POSE_LABEL_NAMES
from tactile_feature_extraction.utils.utils_params import FT_LABEL_NAMES

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_dir(dir):
    # check_dir(dir)
    os.makedirs(dir, exist_ok=True)


def check_dir(dir):

    # check save dir exists
    if os.path.isdir(dir):
        str_input = input("Save Directory already exists, would you like to continue (y,n)? ")
        if not str2bool(str_input):
            exit()
        else:
            # clear out existing files
            empty_dir(dir)


def empty_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_json_obj(obj, name):
    with open(name + ".json", "w") as fp:
        json.dump(obj, fp)


def load_json_obj(name):
    with open(name + ".json", "r") as fp:
        return json.load(fp)


def make_save_dir_str(async_data, task, sensors):
    """
    Combines tasks, input/target dirs and collection modes into single path
    """
    return os.path.join(
        "async" if async_data else "non_async",
        task,
        "_".join([sensor.split('_')[0] for sensor in sensors])
    )


def csv_row_to_label(row):
    return {
            'x': np.array(row['pose_1']),
            'y': np.array(row['pose_2']),
            'z': np.array(row['pose_3']),
            'Rx': np.array(row['pose_4']),
            'Ry': np.array(row['pose_5']),
            'Rz': np.array(row['pose_6']),
            'Fx': np.array(row['Fx']),
            'Fy': np.array(row['Fy']),
            'Fz': np.array(row['Fz']),
            'Tx': np.array(row['Tx']),
            'Ty': np.array(row['Ty']),
            'Tz': np.array(row['Tz']),
        }


def get_ft_pose_limits(data_dirs, save_dir):
    """
     Get limits for poses of data collected, used to encode/decode pose for prediction
     data_dirs is expected to be a list of data directories

     When using more than one data source, limits are taken at the extremes of those used for collection.
    """
    pose_llims, pose_ulims = [], []
    ft_llims, ft_ulims = [], []
    for data_dir in data_dirs:
        pose_params = load_json_obj(os.path.join(data_dir, 'ft_pose_params'))
        pose_llims.append(pose_params['pose_llims'])
        pose_ulims.append(pose_params['pose_ulims'])
        ft_llims.append(pose_params['ft_llims'])
        ft_ulims.append(pose_params['ft_ulims'])

    pose_llims = np.min(pose_llims, axis=0)
    pose_ulims = np.max(pose_ulims, axis=0)
    ft_llims = np.min(ft_llims, axis=0)
    ft_ulims = np.max(ft_ulims, axis=0)

    # save limits
    ft_pose_limits = {
        'pose_llims': list(pose_llims),
        'pose_ulims': list(pose_ulims),
        'ft_llims': list(ft_llims),
        'ft_ulims': list(ft_ulims),
    }

    save_json_obj(ft_pose_limits, os.path.join(save_dir, 'ft_pose_params'))

    return pose_llims, pose_ulims, ft_llims, ft_ulims


class FTPoseEncoder:

    def __init__(self, target_label_names, ft_pose_limits, device):
        self.device = device
        self.target_label_names = target_label_names

        # create tensors for pose limits
        self.pose_llims_np = np.array(ft_pose_limits[0])
        self.pose_ulims_np = np.array(ft_pose_limits[1])
        self.ft_llims_np = np.array(ft_pose_limits[2])
        self.ft_ulims_np = np.array(ft_pose_limits[3])
        self.pose_llims_torch = torch.from_numpy(self.pose_llims_np).float().to(self.device)
        self.pose_ulims_torch = torch.from_numpy(self.pose_ulims_np).float().to(self.device)
        self.ft_llims_torch = torch.from_numpy(self.ft_llims_np).float().to(self.device)
        self.ft_ulims_torch = torch.from_numpy(self.ft_ulims_np).float().to(self.device)

    def encode_label(self, labels_dict):
        """
        Process raw pose data to NN friendly label for prediction.

        From -> {x, y, z, Rx, Ry, Rz, Fx, Fy, Fz}
        To   -> [norm(x), norm(y), norm(z), cos(Rx), sin(Rx), cos(Ry), sin(Ry), cos(Rz), sin(Rz), norm(Fx), norm(Fy), norm(Fz)]
        """
        # encode pose to predictable label
        encoded_pose = []
        for label_name in self.target_label_names:

            # get the target from the dict
            target = labels_dict[label_name].float().to(self.device)

            # normalize pose label within limits
            if label_name in POS_LABEL_NAMES:
                llim = self.pose_llims_torch[POSE_LABEL_NAMES.index(label_name)]
                ulim = self.pose_ulims_torch[POSE_LABEL_NAMES.index(label_name)]
                norm_target = (((target - llim) / (ulim - llim)) * 2) - 1
                encoded_pose.append(norm_target.unsqueeze(dim=1))

            # sine/cosine encoding of angle
            elif label_name in ROT_LABEL_NAMES:
                '''
                ang = target * np.pi/180
                encoded_pose.append(torch.sin(ang).float().to(self.device).unsqueeze(dim=1))
                encoded_pose.append(torch.cos(ang).float().to(self.device).unsqueeze(dim=1))
                '''
                # 角度差修改为直接计算
                if label_name == 'Rx':
                    llim =-23.07850773
                    ulim = 24.76522619
                elif label_name == 'Ry':
                    llim =-24.09146949
                    ulim = 24.41429579
                norm_target = (((target - llim) / (ulim - llim)) * 2) - 1
                encoded_pose.append(norm_target.unsqueeze(dim=1))


            elif label_name in FT_LABEL_NAMES:
                llim = self.ft_llims_torch[FT_LABEL_NAMES.index(label_name)]
                ulim = self.ft_ulims_torch[FT_LABEL_NAMES.index(label_name)]
                norm_target = (((target - llim) / (ulim - llim)) * 2) - 1


                encoded_pose.append(norm_target.unsqueeze(dim=1))

            elif label_name == 'Fxy':
                llim = 0.00054418
                ulim = 5.97044428
                norm_target = (((target - llim) / (ulim - llim)) * 2) - 1
                encoded_pose.append(norm_target.unsqueeze(dim=1))


        # combine targets to make one label tensor
        labels = torch.cat(encoded_pose, 1)


        return labels


    def decode_label(self, outputs):
        """
        Process NN predictions to raw pose data, always decodes to cpu.

        From  -> [norm(x), norm(y), norm(z), cos(Rx), sin(Rx), cos(Ry), sin(Ry), cos(Rz), sin(Rz)]
        To    -> {x, y, z, Rx, Ry, Rz}
        """

        # decode preictable label to pose
        decoded_pose = {
            'x': np.zeros(outputs.shape[0]),
            'y': np.zeros(outputs.shape[0]),
            'z': torch.zeros(outputs.shape[0]),
            'Rx': torch.zeros(outputs.shape[0]),
            'Ry': torch.zeros(outputs.shape[0]),
            'Rz': torch.zeros(outputs.shape[0]),
            'Fx': torch.zeros(outputs.shape[0]),
            'Fy': torch.zeros(outputs.shape[0]),
            'Fz': torch.zeros(outputs.shape[0]),
        }

        label_name_idx = 0
        for label_name in self.target_label_names:

            if label_name in POS_LABEL_NAMES:
                predictions = outputs[:, label_name_idx].detach().cpu()
                llim = self.pose_llims_np[POSE_LABEL_NAMES.index(label_name)]
                ulim = self.pose_ulims_np[POSE_LABEL_NAMES.index(label_name)]
                decoded_predictions = (((predictions + 1) / 2) * (ulim - llim)) + llim
                decoded_pose[label_name] = decoded_predictions
                label_name_idx += 1

            elif label_name in ROT_LABEL_NAMES:
                '''
                sin_predictions = outputs[:, label_name_idx].detach().cpu()
                cos_predictions = outputs[:, label_name_idx + 1].detach().cpu()
                pred_rot = torch.atan2(sin_predictions, cos_predictions)
                pred_rot = pred_rot * (180.0 / np.pi)
                '''
                # 角度差修改为直接计算
                predictions = outputs[:, label_name_idx].detach().cpu()
                if label_name == 'Rx':
                    llim =-23.07850773
                    ulim = 24.76522619
                elif label_name == 'Ry':
                    llim =-24.09146949
                    ulim = 24.41429579
                pred_rot = (((predictions + 1) / 2) * (ulim - llim)) + llim
                decoded_pose[label_name] = pred_rot
                label_name_idx += 1

            elif label_name in FT_LABEL_NAMES:
                predictions = outputs[:, label_name_idx].detach().cpu()
                llim = self.ft_llims_np[FT_LABEL_NAMES.index(label_name)]
                ulim = self.ft_ulims_np[FT_LABEL_NAMES.index(label_name)]
                decoded_predictions = (((predictions + 1) / 2) * (ulim - llim)) + llim
                decoded_pose[label_name] = decoded_predictions
                label_name_idx += 1
            elif label_name == 'Fxy':
                predictions = outputs[:, label_name_idx].detach().cpu()
                llim = 0.00054418
                ulim = 5.97044428
                decoded_predictions = (((predictions + 1) / 2) * (ulim - llim)) + llim
                decoded_pose[label_name] = decoded_predictions
                label_name_idx += 1

        return decoded_pose

    def calc_batch_metrics(self, labels, predictions):
        """
        Calculate metrics useful for measuring progress throughout training.
        """
        err_df = self.err_metric(labels, predictions)
        acc_df = self.acc_metric(err_df)
        return err_df, acc_df

    def err_metric(self, labels, predictions):
        """
        Error metric for regression problem, returns dict of errors in interpretable units.
        Position error (mm), Rotation error (degrees).
        """
        err_df = pd.DataFrame(columns=POSE_LABEL_NAMES)
        for label_name in self.target_label_names:

            if label_name in [*POS_LABEL_NAMES, *FT_LABEL_NAMES,'Fxy']:
                abs_err = torch.abs(
                    labels[label_name] - predictions[label_name]
                ).detach().cpu().numpy()

            elif label_name in ROT_LABEL_NAMES:

                # convert rad
                targ_rot = labels[label_name] * np.pi/180
                pred_rot = predictions[label_name] * np.pi/180

                # Calculate angle difference, taking into account periodicity (thanks ChatGPT)
                abs_err = torch.abs(
                    torch.atan2(torch.sin(targ_rot - pred_rot), torch.cos(targ_rot - pred_rot))
                ).detach().cpu().numpy() * (180.0 / np.pi)

            err_df[label_name] = abs_err

        return err_df

    def mse_metric(self, labels,predictions):
        """
        MSE metric for regression problem, returns dict of errors in interpretable units.
        Position error (mm), Rotation error (degrees).
        """
        mse_df = pd.DataFrame(columns=POSE_LABEL_NAMES)
        for label_name in self.target_label_names:

            if label_name in [*POS_LABEL_NAMES, *FT_LABEL_NAMES,'Fxy']:
                mse_err = ((labels[label_name] - predictions[label_name])**2).detach().cpu().numpy()
            elif label_name in ROT_LABEL_NAMES:

                    # convert rad
                    targ_rot = labels[label_name] * np.pi/180
                    pred_rot = predictions[label_name] * np.pi/180

                    '''
                    # Calculate angle difference, taking into account periodicity (thanks ChatGPT)
                    mse_err = torch.abs(
                        torch.abs(
                            torch.atan2(torch.sin(targ_rot - pred_rot), torch.cos(targ_rot - pred_rot))
                        ).detach().cpu().numpy() * (180.0 / np.pi)
                    )**2
                    '''
                    raise
            mse_df[label_name] = mse_err
        return mse_df

    def r_square_metric(self, labels, predictions):
        """
        R^2 (coefficient of determination) metric for regression problem, returns a dictionary of R^2 scores.
        Position error (mm), Rotation error (degrees).
        """
        r2_dict = {}
        for label_name in self.target_label_names:

            if label_name in [*POSE_LABEL_NAMES, *FT_LABEL_NAMES,'Fxy']:
                # Calculate total sum of squares
                y_true = labels[label_name]
                y_pred = predictions[label_name]
                ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
                # Calculate residual sum of squares
                ss_residual = np.sum((y_true - y_pred) ** 2)
                # Calculate R^2 score
                if ss_total == 0:
                    r2 = 1
                else: r2 = 1 - (ss_residual / ss_total)

            elif label_name in ROT_LABEL_NAMES:

                # Convert degrees to radians
                targ_rot = labels[label_name] * np.pi / 180
                pred_rot = predictions[label_name] * np.pi / 180

                # Calculate angle difference, taking into account periodicity
                angle_diff = torch.atan2(torch.sin(targ_rot - pred_rot), torch.cos(targ_rot - pred_rot))
                angle_diff_np = angle_diff.detach().cpu().numpy() * (180.0 / np.pi)

                # Calculate total sum of squares for angles
                y_true = labels[label_name].detach().cpu().numpy()
                y_true_rad = y_true * np.pi / 180
                y_mean_rad = np.mean(y_true_rad)
                ss_total = np.sum((y_true_rad - y_mean_rad) ** 2) * (180.0 / np.pi) ** 2

                # Calculate residual sum of squares for angles
                ss_residual = np.sum(angle_diff_np ** 2)

                # Calculate R^2 score
                r2 = 1 - (ss_residual / ss_total)

            r2_dict[label_name] = r2
        r2_df = pd.DataFrame([r2_dict])
        return r2_df

    def acc_metric(self, err_df):
        """
        Accuracy metric for regression problem, counting the number of predictions within a tolerance.
        Position Tolerance (mm), Rotation Tolerance (degrees)
        """

        batch_size = err_df.shape[0]
        acc_df = pd.DataFrame(columns=[*POSE_LABEL_NAMES, 'overall_acc'])
        overall_correct = np.ones(batch_size, dtype=bool)
        for label_name in self.target_label_names:

            if label_name in POS_LABEL_NAMES:
                abs_err = err_df[label_name]
                correct = (abs_err < POS_TOL)

            if label_name in ROT_LABEL_NAMES:
                abs_err = err_df[label_name]
                correct = (abs_err < ROT_TOL)

            if label_name in FORCE_LABEL_NAMES:
                abs_err = err_df[label_name]
                correct = (abs_err < FORCE_TOL)

            if label_name in TORQUE_LABEL_NAMES:
                abs_err = err_df[label_name]
                correct = (abs_err < TORQUE_TOL)

            if label_name == 'Fxy':
                abs_err = err_df[label_name]
                correct = (abs_err < FORCE_TOL)

            overall_correct = overall_correct & correct
            acc_df[label_name] = correct.astype(np.float32)

        # count where all predictions are correct for overall accuracy
        acc_df['overall_acc'] = overall_correct.astype(np.float32)

        return acc_df


