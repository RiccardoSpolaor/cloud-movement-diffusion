from typing import List

from fastprogress import progress_bar
import numpy as np
import torch
import torchvision.transforms as T
import wandb
import cv2

# from pathlib import Path

'''def ls(path: Path):
    "Return files on Path, sorted"
    return sorted(list(path.iterdir()))''';

# TOD

class CloudDataset:
    """Dataset for cloud images
    It loads numpy files from wandb artifact and stacks them into a single array
    It also applies some transformations to the images
    """
    def __init__(
        self,
        files: List[str], # list of numpy files to load (they come from the artifact)
        num_frames: int = 4, # how many consecutive frames to stack
        scale: bool = True, # if we images to interval [-0.5, 0.5]
        img_size: int = 64, # resize dim, original images are big (446, 780)
        apply_deterministic_transforms: bool = False, # if True, transforms are deterministic
        ) -> None:
        # Define the transformations to apply to the images.
        if apply_deterministic_transforms:
            transforms = [T.CenterCrop(img_size)]
        else: 
            transforms = [T.RandomCrop(img_size)]
        # Compoe the transformations.
        self.transforms = T.Compose(transforms)
        # Get the data.
        self.data = self.get_data(files, num_frames, scale)

    def _calculate_mean_std(self, array: np.ndarray):
        "Calculate mean and std for normalization"
        mean, std = array.mean(), array.std()
        self.means.append(mean)
        self.stds.append(std)

    @staticmethod
    def _resize_and_transpose(array: np.ndarray, img_size: int) -> np.ndarray:
        # Get the number of events and frames.
        num_events, num_frames = array.shape[0], array.shape[3]
        # Set an empty array to store the resized and transposed array.
        resized_array = np.empty((num_events, img_size, img_size, num_frames))
        # For each event, resize the array to the desired size.
        for event in range(num_events):
            resized_array[event] = cv2.resize(array[event], (img_size, img_size))
        # Transpose the array to shape (num_events, num_frames, img_size, img_size).
        return resized_array.transpose((0, 3, 1, 2))

    def load_channel(self, file_path: str, scale: bool = True):
        one_channel = np.load(file_path)
        one_channel = one_channel.astype(np.float32)
        if scale:
            one_channel = 0.5 - self._scale(one_channel)
            # self._calculate_mean_std(one_channel)
        return one_channel

    def create_windows(self, data, num_frames):
        windows = []
        for event in data:
            wds = np.lib.stride_tricks.sliding_window_view(
                        event,
                        num_frames,
                        axis=0)[::num_frames].transpose(0,4,1,2,3) # (windows, frames, channels, height, width)
            windows.append(wds)
        windows = np.array(windows)
        shape = windows.shape
        windows = windows.reshape(shape[0] * shape[1], shape[2], shape[3], shape[4], shape[5])
        windows = windows.astype('float32')
        return windows # (batch, channels, frames, height, width)

    def get_data(
        self, 
        files: List[str],
        num_frames: int,
        scale: bool,
        img_size: int
        ) -> np.ndarray:
        "Loads all data into a single array self.data"
        channels = []
        # Stack all information channels into a single array.
        for file in progress_bar(files, leave=False):
            # Get a single information channel.
            channel = self.load_channel(file, scale)
            # Resize and transpose the array.
            resized_array = self._resize_and_transpose(channel, img_size)
            # Append the array to the list of channels.
            channels.append(resized_array)
            # TODO: Doesn't garbage collecting work automatically in
            # this case?
            del channel
        # Stack all channels into a single array.
        # Shape = (num_events, num_frames, width, height).
        # TODO: Are we stacking correctly? Shouldn't we stack along the channels axis?
        all_channels = np.stack(channels, axis=2)
        return self.create_windows(all_channels, num_frames)

    def shuffle(self) -> 'CloudDataset':
        """
        Shuffle the dataset in place. This function is useful for
        getting interesting samples on the validation dataset
        
        Returns
        -------
        CloudDataset
            The dataset itself shuffled.
        """
        idxs = torch.randperm(len(self.data))
        self.data = self.data[idxs]
        return self

    @staticmethod
    def _scale(array: np.ndarray) -> np.ndarray:
        """
        Apply min-max scaling to an array in order to get values in
        the interval [0,1].

        Parameters
        ----------
        array : ndarray
            The array to scale.

        Returns
        -------
        ndarray
            The scaled array.
        """
        min, max = array.min(), array.max()
        return (array - min) / (max - min)

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        """
        Get an instance from the dataset at a given index, represented as a
        pytorch tensor. Moreover, the defined transformations are applied to
        the instance.

        Parameters
        ----------
        idx : int
            The index of the instance to get.

        Returns
        -------
        FloatTensor
            The instance at the given index represented as a PyTorch tensor.
        """
        item = self.data[idx]
        item = torch.from_numpy(item)
        return self.transforms(item)

    def __len__(self) -> int:
        """
        Get the number of instances in the dataset.
        
        Returns
        -------
        int
            The number of instances in the dataset.
        """ 
        return len(self.data)

    def save(self, file_name="cloud_frames.npy") -> None:
        """Save the dataset to a numpy file.

        Parameters
        ----------
        file_name : str, optional
            The name of the file to save the dataset to, by default
            "cloud_frames.npy".
        """
        np.save(file_name, self.data)

# TODO: MAKE SURE THAT WHEN DOWNLOADING THE DATASET WE DON'T PUT VALIDATION FRAMES
# IN THE TRAINING DATASET
def download_dataset(at_name, project_name):
    "Downloads dataset from wandb artifact"
    def _get_dataset(run):
        artifact = run.use_artifact(at_name, type='dataset')
        return artifact.download()

    if wandb.run is not None:
        run = wandb.run
        artifact_dir = _get_dataset(run)
    else:
        run = wandb.init(project=project_name, job_type="download_dataset")
        artifact_dir = _get_dataset(run)
        run.finish()

    files = sorted(list(artifact_dir.iterdir()))

    # ls(Path(artifact_dir))
    return files
