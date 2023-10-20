import pickle
from pathlib import Path
from typing import List

from fastprogress import progress_bar
import numpy as np
import torch
import torchvision.transforms as T
import wandb
import cv2
import os
from .scaler import Scaler

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
        scalers: list = None, # list of scalers to apply to the channels
        #scale: bool = True, # if we images to interval [-0.5, 0.5]
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
        self.data = self.get_data(files, num_frames, scalers, img_size)

#    def _calculate_mean_std(self, array: np.ndarray) -> None:
#        """Compute the mean and standard deviation of an array and append them
#        to the corresponding lists.
#
#        Parameters
#        ----------
#        array : np.ndarray
#            The array to compute the mean and standard deviation of.
#        """
#        mean, std = array.mean(), array.std()
#        self.means.append(mean)
#        self.stds.append(std)

    @staticmethod
    def _resize_and_transpose(array: np.ndarray, img_size: int) -> np.ndarray:
        """
        Resize an array to the desired size and transpose it to the
        given image size.
        
        Parameters
        ----------
        array : np.ndarray
            The array to resize and transpose.
        img_size : int
            The size of the image to transpose the array to.
        """
        # Get the number of events and frames.
        num_events, num_frames = array.shape[0], array.shape[3]
        # Set an empty array to store the resized and transposed array.
        resized_array = np.empty((num_events, img_size, img_size, num_frames))
        # For each event, resize the array to the desired size.
        for event in range(num_events):
            resized_array[event] = cv2.resize(array[event], (img_size, img_size))
        # Transpose the array to shape (num_events, num_frames, img_size, img_size).
        return resized_array.transpose((0, 3, 1, 2))

    def load_channel(self, file_path: str, scaler: object) -> np.ndarray:
        """Load a single channel from a numpy file and apply min-max
        scaling to it.

        Parameters
        ----------
        file_path : str
            The file path of the numpy file to load.
        scale : bool, optional
            Whether to scale or not the channel through min-max,
            by default True.

        Returns
        -------
        ndarray
            The scaled channel.
        """
        one_channel = np.load(file_path)
        one_channel = one_channel.astype(np.float32)
        if scaler:
            one_channel = scaler.scale(one_channel)
            # self._calculate_mean_std(one_channel)
        return one_channel

    def create_windows(self, data: np.ndarray, num_frames: int) -> np.ndarray:
        """Create windows of consecutive frames from the data.

        Parameters
        ----------
        data : ndarray
            The data to create windows from.
        num_frames : int
            The number of consecutive frames to stack.

        Returns
        -------
        ndarray
            The windows of consecutive frames.
        """
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
        scalers: list,
        img_size: int
        ) -> np.ndarray:
        """
        Load all data channels from the given files and stack them
        into a single array.
        
        Parameters
        ----------
        files : list of str
            The list of files to load.
        num_frames : int
            The number of consecutive frames to stack.
        scale : bool
            Whether to scale or not the channels through min-max.
        img_size : int
            The size of the image to transpose the array to.
        """
        channels = []
        # Stack all information channels into a single array.
        for i, file in enumerate(progress_bar(files, leave=False)):
            # Get a single information channel.
            if scalers is not None:
                channel = self.load_channel(file, scalers[i])
            else:
                channel = self.load_channel(file, None)
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


        # channels[-2:] = [np.concatenate(channels[-2:], axis=0)] # counter the split of big files

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

    '''@staticmethod
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
        return (array - min) / (max - min)''';

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


def inspect_data(file_path: List[Path]) -> None:
    """Inspect the data in the given file path for big files and split them.

    Parameters  
    ----------
    file_path : List[Path]
        The list of files to inspect.
    Returns
    -------
    files : List[Path]
    Refined file path

    """
    files = file_path
    for file in files:
        if (data := np.load(file)).size > 1000000000 :
            chunks = np.split(data, 2, axis=0)
            for i,c in enumerate(chunks):
                #print("chunk shape:", c.shape)
                np.save(str(file)[:-4]+f'_{i}.npy',c)
            os.remove(file)
            files.remove(file)
            del data
    return files

# TODO: MAKE SURE THAT WHEN DOWNLOADING THE DATASET WE DON'T PUT VALIDATION FRAMES
# IN THE TRAINING DATASET
def download_dataset(at_name: str, project_name: str) -> List[str]:
    """Download the dataset from wandb.

    Parameters
    ----------
    at_name : str
        The name of the artifact to download.
    project_name : str
        The name of the project to download the artifact from.
    
    Returns
    -------
    list of str
        The list of files in the downloaded dataset.
    """
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

    files = sorted(list(Path(artifact_dir).iterdir()))
    return files #inspect_data(files)

def download_scalers(at_name: str, project_name: str) -> List[Scaler]:
    """Download the scaler objects from wandb.

    Parameters
    ----------
    at_name : str
        The name of the artifact to download.
    project_name : str
        The name of the project to download the artifact from.
    
    Returns
    -------
    list
        The list of scaler objects downloaded from wandb.
    """
    def _get_dataset(run):
        artifact = run.use_artifact(at_name, type='pickle')
        return artifact.download()

    if wandb.run is not None:
        run = wandb.run
        artifact_dir = _get_dataset(run)
    else:
        run = wandb.init(project=project_name, job_type='download_scalers')
        artifact_dir = _get_dataset(run)
        run.finish()

    files = sorted(list(Path(artifact_dir).iterdir()))
    scalers = []

    for f in files:
        with open(f, 'rb') as file:
            scaler = pickle.load(file)
            scalers.append(scaler)
    
    return scalers
