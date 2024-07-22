import os
import argparse

from tactile_feature_extraction.utils.utils_learning import save_json_obj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--async_data', action='store_true')
    parser.add_argument('--no_async_data', dest='async_data', action='store_false')
    parser.set_defaults(async_data=False)
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['linshear_surface_3d'].",
        default=['linshear_surface_3d']
    )
    parser.add_argument(
        '-s', '--sensors',
        nargs='+',
        help="Choose sensor from ['s1_minitip_331', 's2_minitip_331', 's3_minitip_331'].",
        default=['331']
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose model from ['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit'].",
        default=['simple_cnn']
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cuda'
    )

    # parse arguments
    args = parser.parse_args()
    return args


def setup_learning(save_dir=None):

    # Parameters
    learning_params = {
        'seed': 42,
        'batch_size': 16,
        'epochs': 10, # 100
        'lr': 1e-5, # 1e-4
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 1e-6,
        'adam_b1': 0.9,
        'adam_b2': 0.999,
        'shuffle': True,
        'n_cpu': 11,
    }

    image_processing_params = {
        'dims': (128, 128),
        'bbox': None,
        # 'blur': 11,
        'blur': None,
        # 'thresh': [55, -2],
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    frame_processing_params = {
        'n_stack': 2,
        **image_processing_params
    }

    augmentation_params = {
        'rshift': None,
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    if save_dir:
        save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
        save_json_obj(image_processing_params, os.path.join(save_dir, 'image_processing_params'))
        save_json_obj(frame_processing_params, os.path.join(save_dir, 'frame_processing_params'))
        save_json_obj(augmentation_params, os.path.join(save_dir, 'augmentation_params'))

    return learning_params, image_processing_params, frame_processing_params, augmentation_params


def setup_model(model_type, save_dir):

    model_params = {
        'model_type': model_type
    }

    if model_type == 'simple_cnn':
        model_params['model_kwargs'] = {
                'conv_layers': [32, 32, 32, 32],
                'conv_kernel_sizes': [11, 9, 7, 5],
                'fc_layers': [512, 512],
                'activation': 'relu',
                'dropout': 0.0,
                'apply_batchnorm': True,
        }

    elif model_type == 'posenet_cnn':
        model_params['model_kwargs'] = {
                'conv_layers': [256, 256, 256, 256, 256],
                'conv_kernel_sizes': [3, 3, 3, 3, 3],
                'fc_layers': [64],
                'activation': 'elu',
                'dropout': 0.0,
                'apply_batchnorm': True,
        }

    elif model_type == 'nature_cnn':
        model_params['model_kwargs'] = {
            'fc_layers': [512, 512],
            'dropout': 0.0,
        }

    elif model_type == 'resnet':
        model_params['model_kwargs'] = {
            'layers': [2, 2, 2, 2],
        }

    elif model_type == 'vit':
        model_params['model_kwargs'] = {
            'patch_size': 32,
            'dim': 128,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 512,
            'pool': 'mean',  # for regression
        }
    elif model_type == 'conv_lstm':
        model_params['model_kwargs'] = {
            'conv_layers': [32, 32, 32, 32],
            'conv_kernel_sizes': [11, 9, 7, 5],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }
    elif model_type == 'conv_transformer':
        model_params['model_kwargs'] = {
            'conv_layers': [32, 32, 32, 32],
            'conv_kernel_sizes': [11, 9, 7, 5],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }
    elif model_type == 'conv_gru':
        model_params['model_kwargs'] = {
            'conv_layers': [32, 32, 32, 32],
            'conv_kernel_sizes': [11, 9, 7, 5],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }
    elif model_type == 'seq2seq_gru':
        model_params['model_kwargs'] = {
            'conv_layers': [32, 32, 32, 32],
            'conv_kernel_sizes': [11, 9, 7, 5],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }
    elif model_type == 'CNN3D':
        model_params['model_kwargs'] = {
            'conv_layers': [32, 32, 32, 32],
            'conv_kernel_sizes': [(3, 11, 11), (3, 9, 9), (3, 7, 7), (3, 5, 5)],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }
    elif model_type == 'seq2seq_gru_attention':
        model_params['model_kwargs'] = {
            'conv_layers': [32, 32, 32, 32],
            'conv_kernel_sizes': [11, 9, 7, 5],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }

    # save parameters
    save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params


def setup_task(task_name):
    """
    Returns task specific details.
    """

    if task_name == 'linshear_surface_3d':
        out_dim = 6
        label_names = ['z', 'Rx', 'Ry', 'Fx', 'Fy', 'Fz']
    elif task_name == 'only_ft':
        out_dim = 3
        label_names = ['Fx', 'Fy', 'Fz']

    else:
        raise ValueError('Incorrect task_name specified: {}'.format(task_name))

    return out_dim, label_names
