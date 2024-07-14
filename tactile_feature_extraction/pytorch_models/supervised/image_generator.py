import numpy as np
import os
import cv2
import pandas as pd
import torch
import json
import pickle

from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.image_transforms import augment_image


class ImageDataGenerator(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dirs,
        csv_row_to_label,
        dims=(128, 128),
        bbox=None,
        stdiz=False,
        normlz=False,
        blur=None,
        thresh=None,
        rshift=None,
        rzoom=None,
        brightlims=None,
        noise_var=None,
    ):

        # check if data dirs are lists
        assert isinstance(data_dirs, list), "data_dirs should be a list!"

        self._dims = dims
        self._bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._blur = blur
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var

        self._csv_row_to_label = csv_row_to_label

        # load csv file
        self._label_df = self.load_data_dirs(data_dirs)

    def load_data_dirs(self, data_dirs):

        # add collumn for which dir data is stored in
        df_list = []
        for data_dir in data_dirs:
            df = pd.read_csv(os.path.join(data_dir, 'targets.csv'))

            # check for a processed image dir first
            image_dir = os.path.join(data_dir, 'processed_images')

            # fall back on standard images
            if not os.path.isdir(image_dir):
                image_dir = os.path.join(data_dir, 'images')

            df['image_dir'] = image_dir
            df_list.append(df)

        # concat all df
        full_df = pd.concat(df_list)

        return full_df

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        row = self._label_df.iloc[index]
        image_filename = os.path.join(row['image_dir'], row['sensor_image'])
        raw_image = cv2.imread(image_filename)

        # preprocess/augment image
        processed_image = process_image(
            raw_image,
            gray=True,
            bbox=self._bbox,
            dims=self._dims,
            stdiz=self._stdiz,
            normlz=self._normlz,
            blur=self._blur,
            thresh=self._thresh,
        )

        processed_image = augment_image(
            processed_image,
            rshift=self._rshift,
            rzoom=self._rzoom,
            brightlims=self._brightlims,
            noise_var=self._noise_var
        )

        # put the channel into first axis because pytorch
        processed_image = np.rollaxis(processed_image, 2, 0)

        # get label
        target = self._csv_row_to_label(row)
        sample = {'images': processed_image, 'labels': target}

        return sample

class feature_data_generator(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs,
        csv_row_to_label,
        dims=(128, 128),
        bbox=None,
        stdiz=False,
        normlz=False,
        blur=None,
        thresh=None,
        rshift=None,
        rzoom=None,
        brightlims=None,
        noise_var=None,
    ):
        # check if data dirs are lists
        assert isinstance(data_dirs, list), "data_dirs should be a list!"

        self._csv_row_to_label = csv_row_to_label

        # load csv file
        self._label_df = self.load_data_dirs(data_dirs)

    def load_data_dirs(self, data_dirs):
        # add column for which dir data is stored in
        df_list = []
        for data_dir in data_dirs:
            df = pd.read_csv(os.path.join(data_dir, 'targets.csv'))

            # check for a processed npy dir first
            feature_dir = os.path.join(data_dir, 'processed_features')

            # fall back on standard npy files
            if not os.path.isdir(feature_dir):
                feature_dir = os.path.join(data_dir, 'features')

            df['feature_dir'] = feature_dir
            df_list.append(df)

        # concat all df
        full_df = pd.concat(df_list)
        return full_df

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        row = self._label_df.iloc[index]
        feature_filename = os.path.join(row['feature_dir'], row['sensor_features'])
        features = np.load(feature_filename)

        num_frames = 100
        if features.shape[0] > num_frames:
            features = features[-num_frames:, :]
        elif features.shape[0] < num_frames:
            pad_size = num_frames - features.shape[0]
            padding = np.tile(features[0, :], (pad_size, 1))
            features = np.vstack((padding, features))


        # get label
        target = self._csv_row_to_label(row)
        sample = {'images': torch.tensor(features, dtype=torch.float32), 'labels': target}

        return sample

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir, labels_dir, frame_indices, n_frames, transform=None):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.n_frames = n_frames
        self.transform = transform
        self.frame_indices = json.load(open(frame_indices))
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        feature_files = [f for f in os.listdir(self.features_dir) if f.endswith('_features.npy')]
        for feature_file in feature_files:
            feature_path = os.path.join(self.features_dir, feature_file)
            features = np.load(feature_path)
            indices = self.frame_indices[feature_file.replace('_features.npy', '')]
            label_file = feature_file.replace('_features.npy', '.pkl')
            label_path = os.path.join(self.labels_dir, label_file)

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"Label file {label_path} not found. Skipping.")
                continue

            with open(label_path, 'rb') as f:
                labels = pickle.load(f)

            # 从标签字典中提取帧编号和对应的Fx, Fy, Fz值
            frames = labels['frame']
            fx_values = labels['fx']
            fy_values = labels['fy']
            fz_values = labels['fz']
            num_frames = len(frames)
            label_index = 0
            for i in range(num_frames):
                input_frames = []
                frame_index = frames[i]
                '''
                if i==0:
                    frame_index = frames[i]
                elif frames[i] == frame_index+1: # 检查帧是否连续
                    frame_index = frames[i] # 获取当前帧的索引
                else:
                    label_index = 0
                    frame_index = frames[i]
                '''  # 如果不考虑label的跳跃，一般来说问题不大，但是存在一种可能，label序号跳跃了，frame也跳跃了。如果这样的话，会报错。
                if frame_index not in indices:
                    label_index = 0
                    # frame_index = frames[i]
                    continue

                for j in range(self.n_frames):
                    if label_index - j < 0:
                        input_frames.insert(0, features[indices.index(frames[i - label_index])])
                    else:
                        input_frames.insert(0, features[indices.index(frame_index - j)])

                input_frames = np.stack(input_frames)
                label = [fx_values[i], fy_values[i], fz_values[i]]  # 获取当前帧的标签
                samples.append((input_frames, label))
                label_index += 1
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_data, label = self.samples[idx]

        if self.transform:
            frames_data = self.transform(frames_data)
        # return torch.tensor(frames_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        # labels为dataframe，第一个值标为fx，第二个值标为fy，第三个值标为fz
        labels = {'Fx': label[0], 'Fy': label[1], 'Fz': label[2]}
        sample = {'images': torch.tensor(frames_data, dtype=torch.float32), 'labels': labels}
        return sample


class PhotoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            photos_dir,
            labels_dir,
            n_frames,
            dims=(128, 128),
            bbox=None,
            stdiz=False,
            normlz=False,
            blur=None,
            thresh=None,
            rshift=None,
            rzoom=None,
            brightlims=None,
            noise_var=None,
    ):
        self.photos_dir = photos_dir
        self.labels_dir = labels_dir
        self.n_frames = n_frames

        self._dims = dims
        self._bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._blur = blur
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var

        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        videos = [f for f in os.listdir(self.photos_dir)]
        for video in videos:
            video_path = os.path.join(self.photos_dir, video)
            label_file = video + '.pkl'
            label_path = os.path.join(self.labels_dir, label_file)

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"Label file {label_path} not found. Skipping.")
                continue

            with open(label_path, 'rb') as f:
                labels = pickle.load(f)

            # 从标签字典中提取帧编号和对应的Fx, Fy, Fz值
            frames = labels['frame']
            fx_values = labels['fx']
            fy_values = labels['fy']
            fz_values = labels['fz']
            num_frames = len(frames)
            for i in range(num_frames):
                frame_index = frames[i]
                photo_path = video_path + '/frame_' + str(frame_index) + '.png'
                if not os.path.exists(photo_path):
                    continue
                raw_image = cv2.imread(photo_path)

                # preprocess/augment image
                processed_image = process_image(
                    raw_image,
                    gray=True,
                    bbox=self._bbox,
                    dims=self._dims,
                    stdiz=self._stdiz,
                    normlz=self._normlz,
                    blur=self._blur,
                    thresh=self._thresh,
                )

                processed_image = augment_image(
                    processed_image,
                    rshift=self._rshift,
                    rzoom=self._rzoom,
                    brightlims=self._brightlims,
                    noise_var=self._noise_var
                )

                # put the channel into first axis because pytorch
                processed_image = np.rollaxis(processed_image, 2, 0)
                label = [fx_values[i], fy_values[i], fz_values[i]]
                samples.append((processed_image, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_data, label = self.samples[idx]

        labels = {'Fx': label[0], 'Fy': label[1], 'Fz': label[2]}
        sample = {'images': torch.tensor(frames_data, dtype=torch.float32), 'labels': labels}
        return sample

from tqdm import tqdm
class PhotoDataset_ConvLstm(torch.utils.data.Dataset):
    def __init__(
            self,
            photos_dir,
            labels_dir,
            n_frames,
            padding=True,
            dims=(128, 128),
            bbox=None,
            stdiz=False,
            normlz=False,
            blur=None,
            thresh=None,
            rshift=None,
            rzoom=None,
            brightlims=None,
            noise_var=None,
    ):
        self.photos_dir = photos_dir
        self.labels_dir = labels_dir
        self.n_frames = n_frames
        self.padding = padding

        self._dims = dims
        self._bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._blur = blur
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var

        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        videos = [f for f in os.listdir(self.photos_dir)]
        for video in tqdm(videos):
            video_path = os.path.join(self.photos_dir, video)
            label_file = video + '.pkl'
            label_path = os.path.join(self.labels_dir, label_file)

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"Label file {label_path} not found. Skipping.")
                continue

            with open(label_path, 'rb') as f:
                labels = pickle.load(f)

            # 从标签字典中提取帧编号和对应的Fx, Fy, Fz值
            frames = labels['frame']
            fx_values = labels['fx']
            fy_values = labels['fy']
            fz_values = labels['fz']
            num_frames = len(frames)
            label_index = 0
            for i in range(num_frames):
                input_photos_dirs = []
                frame_index = frames[i]
                photo_path = video_path + '/frame_' + str(frame_index) + '.png'
                if not os.path.exists(photo_path):
                    label_index = 0
                    continue
                for j in range(self.n_frames):
                    if label_index - j < 0:
                        if self.padding:
                            input_photos_dirs.insert(0, video_path + '/frame_' + str(frames[i - label_index]) + '.png')
                        else:
                            continue
                    else:
                        input_photos_dirs.insert(0, video_path + '/frame_' + str(frame_index - j) + '.png')
                input_photos= []
                for photo_path in input_photos_dirs:
                    photo= cv2.imread(photo_path)
                    # preprocess/augment image
                    processed_image = process_image(
                        photo,
                        gray=True,
                        bbox=self._bbox,
                        dims=self._dims,
                        stdiz=self._stdiz,
                        normlz=self._normlz,
                        blur=self._blur,
                        thresh=self._thresh,
                    )
                    processed_image = augment_image(
                        processed_image,
                        rshift=self._rshift,
                        rzoom=self._rzoom,
                        brightlims=self._brightlims,
                        noise_var=self._noise_var
                    )
                    # put the channel into first axis because pytorch
                    processed_image = np.rollaxis(processed_image, 2, 0)
                    input_photos.append(processed_image)
                input_photos = np.stack(input_photos) # shape = (n_frames, 1, width, height)
                label = [fx_values[i], fy_values[i], fz_values[i]]
                samples.append((input_photos, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_data, label = self.samples[idx]

        labels = {'Fx': label[0], 'Fy': label[1], 'Fz': label[2]}
        sample = {'images': torch.tensor(frames_data, dtype=torch.float32), 'labels': labels}
        return sample

class PhotoDataset_Seq2Seq(torch.utils.data.Dataset):
    def __init__(
            self,
            photos_dir,
            labels_dir,
            n_frames,
            padding=True,
            dims=(128, 128),
            bbox=None,
            stdiz=False,
            normlz=False,
            blur=None,
            thresh=None,
            rshift=None,
            rzoom=None,
            brightlims=None,
            noise_var=None,
    ):
        self.photos_dir = photos_dir
        self.labels_dir = labels_dir
        self.n_frames = n_frames
        self.padding = padding

        self._dims = dims
        self._bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._blur = blur
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var

        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        videos = [f for f in os.listdir(self.photos_dir)]
        for video in tqdm(videos):
            video_path = os.path.join(self.photos_dir, video)
            label_file = video + '.pkl'
            label_path = os.path.join(self.labels_dir, label_file)

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"Label file {label_path} not found. Skipping.")
                continue

            with open(label_path, 'rb') as f:
                labels = pickle.load(f)

            # 从标签字典中提取帧编号和对应的Fx, Fy, Fz值
            frames = labels['frame']

            fx_values = labels['fx']
            fy_values = labels['fy']
            fz_values = labels['fz']
            num_frames = len(frames)
            label_index = 0

            skip_labels = []

            for i in range(num_frames):
                input_photos_dirs = []


                if i == 0:
                    frame_index = frames[i]
                elif frames[i] == frame_index + 1:  # 检查帧是否连续
                    frame_index = frames[i]  # 获取当前帧的索引
                else:
                    for skip_num in range(frame_index+1, frames[i]):
                        skip_labels.append(skip_num)
                        # 例如，如果当前帧是10，下一帧是12，那么11这一帧就是缺失的帧
                    frame_index = frames[i]

                photo_path = video_path + '/frame_' + str(frame_index) + '.png'
                labels = []
                if not os.path.exists(photo_path):
                    label_index = 0
                    continue

                for j in range(self.n_frames):
                    if label_index - j < 0:
                        if self.padding:
                            input_photos_dirs.insert(0, video_path + '/frame_' + str(frames[i - label_index]) + '.png')
                            labels.insert(0, [fx_values[i - label_index], fy_values[i - label_index], fz_values[i - label_index]])
                        else:
                            continue
                    else:
                        input_photos_dirs.insert(0, video_path + '/frame_' + str(frame_index - j) + '.png')
                        labels.insert(0, [fx_values[i - j], fy_values[i - j], fz_values[i - j]])
                input_photos= []
                for photo_path in input_photos_dirs:
                    photo= cv2.imread(photo_path)
                    # preprocess/augment image
                    processed_image = process_image(
                        photo,
                        gray=True,
                        bbox=self._bbox,
                        dims=self._dims,
                        stdiz=self._stdiz,
                        normlz=self._normlz,
                        blur=self._blur,
                        thresh=self._thresh,
                    )
                    processed_image = augment_image(
                        processed_image,
                        rshift=self._rshift,
                        rzoom=self._rzoom,
                        brightlims=self._brightlims,
                        noise_var=self._noise_var
                    )
                    # put the channel into first axis because pytorch
                    processed_image = np.rollaxis(processed_image, 2, 0)
                    input_photos.append(processed_image)
                input_photos = np.stack(input_photos) # shape = (n_frames, 1, width, height)
                labels = np.array(labels) # shape = (n_frames, 3)
                samples.append((input_photos, labels))

                if skip_labels!=[]:
                    print(skip_labels)
                    print(video_path)
                    skip_labels = []

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_data, label = self.samples[idx]

        labels = {'Fx': label[0], 'Fy': label[1], 'Fz': label[2]}
        sample = {'images': torch.tensor(frames_data, dtype=torch.float32), 'labels': labels}
        return sample

class PhotoDataset_ConvLstm_2(torch.utils.data.Dataset):
    def __init__(
            self,
            photos_dir,
            labels_dir,
            n_frames,
            padding=True,
            dims=(128, 128),
            bbox=None,
            stdiz=False,
            normlz=False,
            blur=None,
            thresh=None,
            rshift=None,
            rzoom=None,
            brightlims=None,
            noise_var=None,
    ):
        self.photos_dir = photos_dir
        self.labels_dir = labels_dir
        self.n_frames = n_frames
        self.padding = padding

        self._dims = dims
        self._bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._blur = blur
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var

        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        videos = [f for f in os.listdir(self.photos_dir)]
        for video in tqdm(videos):
            video_path = os.path.join(self.photos_dir, video)
            label_file = video + '.pkl'
            label_path = os.path.join(self.labels_dir, label_file)

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"Label file {label_path} not found. Skipping.")
                continue

            with open(label_path, 'rb') as f:
                labels = pickle.load(f)

            # 从标签字典中提取帧编号和对应的Fx, Fy, Fz值
            frames = labels['frame']
            fx_values = labels['fx']
            fy_values = labels['fy']
            fz_values = labels['fz']
            num_frames = len(frames)
            label_index = 0
            for i in range(num_frames):
                input_photos_dirs = []
                frame_index = frames[i]
                photo_path = video_path + '/frame_' + str(frame_index) + '.png'
                if not os.path.exists(photo_path):
                    label_index = 0
                    continue
                for j in range(self.n_frames):
                    if label_index - j < 0:
                        if self.padding:
                            input_photos_dirs.insert(0, video_path + '/frame_' + str(frames[i - label_index]) + '.png')
                        else:
                            continue
                    else:
                        input_photos_dirs.insert(0, video_path + '/frame_' + str(frame_index - j) + '.png')
                label = [fx_values[i], fy_values[i], fz_values[i]]
                samples.append((input_photos_dirs, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_photos_dirs, label = self.samples[idx]
        input_photos = []
        for photo_path in input_photos_dirs:
            photo = cv2.imread(photo_path)
            # preprocess/augment image
            processed_image = process_image(
                photo,
                gray=True,
                bbox=self._bbox,
                dims=self._dims,
                stdiz=self._stdiz,
                normlz=self._normlz,
                blur=self._blur,
                thresh=self._thresh,
            )
            processed_image = augment_image(
                processed_image,
                rshift=self._rshift,
                rzoom=self._rzoom,
                brightlims=self._brightlims,
                noise_var=self._noise_var
            )
            # put the channel into first axis because pytorch
            processed_image = np.rollaxis(processed_image, 2, 0)
            input_photos.append(processed_image)

        input_photos = np.stack(input_photos)
        labels = {'Fx': label[0], 'Fy': label[1], 'Fz': label[2]}
        sample = {'images': torch.tensor(input_photos, dtype=torch.float32), 'labels': labels}
        return sample

def numpy_collate(batch):
    '''
    Batch is list of len: batch_size
    Each element is dict {images: ..., labels: ...}
    Use Collate fn to ensure they are returned as np arrays.
    '''
    # list of arrays -> stacked into array
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)

    # list of lists/tuples -> recursive on each element
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    # list of dicts -> recursive returned as dict with same keys
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}

    # list of non array element -> list of arrays
    else:
        return np.array(batch)


def demo_image_generation(
    data_dirs,
    csv_row_to_label,
    learning_params,
    image_processing_params,
    augmentation_params
):

    # Configure dataloaders
    generator_args = {**image_processing_params, **augmentation_params}
    generator = ImageDataGenerator(
        data_dirs=data_dirs,
        csv_row_to_label=csv_row_to_label,
        **generator_args
    )

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu'],
        collate_fn=numpy_collate
    )

    # iterate through
    for (i_batch, sample_batched) in enumerate(loader, 0):

        # shape = (batch, n_frames, width, height)
        images = sample_batched['images']
        labels = sample_batched['labels']
        cv2.namedWindow("example_images")

        for i in range(learning_params['batch_size']):
            for key, item in labels.items():
                print(key.split('_')[0], ': ', item[i])
            print('')

            # convert image to opencv format, not pytorch
            image = np.moveaxis(images[i], 0, -1)
            cv2.imshow("example_images", image)
            k = cv2.waitKey(500)
            if k == 27:    # Esc key to stop
                exit()
