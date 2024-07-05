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