# TODO: repeated code just do the scaling here.
import numpy as np

class Scaler():
    def __init__(self, dataset: np.ndarray) -> None:
        self.min = dataset.min()
        self.max = dataset.max()

    def scale(self, array: np.ndarray) -> np.ndarray:
        return .5 - (array - self.min) / (self.max - self.min)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        return ((.5 - array) * (self.max - self.min)) + self.min