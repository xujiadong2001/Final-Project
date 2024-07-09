'''
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.image_transforms import augment_image

# 加载预训练的ResNet50模型，并去掉最后的分类层
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features_from_folder(folder_path):
    features = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)  # (169, 225, 3)
            image = preprocess(image)  # torch.Size([3, 224, 298])
            # image = process_image(image, gray=False, bbox=None, dims=(128, 128), stdiz=False, normlz=True, blur=None, thresh=None) # 没有像原有代码一样转为灰度图，不知道转灰度是不是更好
            # image = augment_image(image, rshift=None, rzoom=None, brightlims=None, noise_var=None) # (128, 128, 1)
            # image = np.rollaxis(image, 2, 0) # (1, 128, 128)
            # image = torch.tensor(image).float()
            image = image.unsqueeze(0)  # 添加批次维度
            with torch.no_grad():
                feature = resnet(image)
            features.append(feature.squeeze().numpy())
    return np.array(features)


def process_video_folder(video_folder_path, output_file):
    features = extract_features_from_folder(video_folder_path)
    np.save(output_file, features)
    print(f"Features saved for {os.path.basename(video_folder_path)} in {output_file}")


def process_videos_folder(videos_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for video_folder in os.listdir(videos_folder):
            video_folder_path = os.path.join(videos_folder, video_folder)
            if os.path.isdir(video_folder_path):
                output_file = os.path.join(output_folder, f"{video_folder}_features.npy")
                futures.append(executor.submit(process_video_folder, video_folder_path, output_file))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')


videos_folder = 'collect_331_5D_surface/videos'
output_folder = 'collect_331_5D_surface/features'

process_videos_folder(videos_folder, output_folder)
'''
'''
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.image_transforms import augment_image

# 加载预训练的ResNet50模型，并去掉最后的分类层
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features_from_folder(folder_path):
    features = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path) # (169, 225, 3)
            image = preprocess(image) # torch.Size([3, 224, 298])
            # image = process_image(image, gray=False, bbox=None, dims=(128, 128), stdiz=False, normlz=True, blur=None, thresh=None) # 没有像原有代码一样转为灰度图，不知道转灰度是不是更好
            # image = augment_image(image, rshift=None, rzoom=None, brightlims=None, noise_var=None) # (128, 128, 1)
            # image = np.rollaxis(image, 2, 0) # (1, 128, 128)
            # image = torch.tensor(image).float()
            image = image.unsqueeze(0)  # 添加批次维度
            with torch.no_grad():
                feature = resnet(image)
            features.append(feature.squeeze().numpy())
    return np.array(features)


def process_videos_folder(videos_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_folder in os.listdir(videos_folder):
        video_folder_path = os.path.join(videos_folder, video_folder)
        if os.path.isdir(video_folder_path):
            features = extract_features_from_folder(video_folder_path)
            output_file = os.path.join(output_folder, f"{video_folder}_features.npy")
            np.save(output_file, features)
            print(f"Features saved for {video_folder} in {output_file}")


videos_folder = 'collect_331_5D_surface/videos'
output_folder = 'collect_331_5D_surface/features'

process_videos_folder(videos_folder, output_folder)
'''

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.image_transforms import augment_image

class CNN(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        conv_layers=[16, 16, 16],
        conv_kernel_sizes=[5, 5, 5],
        fc_layers=[128, 128],
        activation='relu',
        apply_batchnorm=False,
        dropout=0.0,
    ):
        super(CNN, self).__init__()

        assert len(conv_layers) > 0, "conv_layers must contain values"
        assert len(fc_layers) > 0, "fc_layers must contain values"
        assert len(conv_layers) == len(conv_kernel_sizes), "conv_layers must be same len as conv_kernel_sizes"

        # add first layer to network
        cnn_modules = []
        cnn_modules.append(nn.Conv2d(in_channels, conv_layers[0], kernel_size=conv_kernel_sizes[0], stride=1, padding=2))
        if apply_batchnorm:
            cnn_modules.append(nn.BatchNorm2d(conv_layers[0]))
        cnn_modules.append(nn.ReLU())
        cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # add the remaining conv layers by iterating through params
        for idx in range(len(conv_layers) - 1):
            cnn_modules.append(
                nn.Conv2d(
                    conv_layers[idx],
                    conv_layers[idx + 1],
                    kernel_size=conv_kernel_sizes[idx + 1],
                    stride=1, padding=2)
                )

            if apply_batchnorm:
                cnn_modules.append(nn.BatchNorm2d(conv_layers[idx+1]))

            if activation == 'relu':
                cnn_modules.append(nn.ReLU())
            elif activation == 'elu':
                cnn_modules.append(nn.ELU())
            cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # create cnn component of network
        self.cnn = nn.Sequential(*cnn_modules)

        # compute shape out of cnn by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, in_channels, *in_dim))
            n_flatten = np.prod(self.cnn(dummy_input).shape)

        fc_modules = []
        fc_modules.append(nn.Linear(n_flatten, fc_layers[0]))
        fc_modules.append(nn.ReLU())
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            if activation == 'relu':
                fc_modules.append(nn.ReLU())
            elif activation == 'elu':
                fc_modules.append(nn.ELU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


# 定义函数移除最后一层并提取特征
def remove_last_layer_and_extract_features(model, input_data):


    with torch.no_grad():
        features = model(input_data)
    return features


# 加载保存的模型
model_path = 'best_model.pth'
model = CNN(in_dim=(128, 128), in_channels=1, out_dim=3,
            conv_layers=[32, 32, 32, 32],
            conv_kernel_sizes=[11, 9, 7, 5],
            fc_layers=[512, 512],
            activation='relu',
            apply_batchnorm=True,
            dropout=0.0)

model.load_state_dict(torch.load(model_path))
# 移除最后一层
model.fc = nn.Sequential(*list(model.fc.children())[:-1])
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # 调整到与模型输入一致的尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features_from_folder(folder_path):
    features = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)  # 读取图像
            image = process_image(image, gray=True, bbox=None, dims=(128, 128), stdiz=False, normlz=True, blur=None, thresh=None)
            image = augment_image(image, rshift=None, rzoom=None, brightlims=None, noise_var=None) # (128, 128, 1)
            image = np.rollaxis(image, 2, 0) # (1, 128, 128)
            image = torch.tensor(image).float()
            image=image.unsqueeze(0)
            with torch.no_grad():
                feature = remove_last_layer_and_extract_features(model, image)
            features.append(feature.squeeze().numpy())
    return np.array(features)


def process_video_folder(video_folder_path, output_file):
    features = extract_features_from_folder(video_folder_path)
    np.save(output_file, features)
    print(f"Features saved for {os.path.basename(video_folder_path)} in {output_file}")


def process_videos_folder(videos_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for video_folder in os.listdir(videos_folder):
            video_folder_path = os.path.join(videos_folder, video_folder)
            if os.path.isdir(video_folder_path):
                output_file = os.path.join(output_folder, f"{video_folder}_features.npy")
                futures.append(executor.submit(process_video_folder, video_folder_path, output_file))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')


videos_folder = 'collect_331_5D_surface/videos'
output_folder = 'collect_331_5D_surface/features_2'

process_videos_folder(videos_folder, output_folder)
