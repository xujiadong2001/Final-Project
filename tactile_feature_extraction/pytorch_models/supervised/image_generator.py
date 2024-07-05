import numpy as np
import os
import cv2
import pandas as pd
import torch

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
