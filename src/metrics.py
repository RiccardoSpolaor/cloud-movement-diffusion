from typing import List
import torch
from torch import nn

class mCSI(nn.Module):
    """
    Class to compute the mean Critical Success Index (mCSI) score.
    
    Attributes
    ----------
    thresholds : list of float
        The thresholds to apply to the target and prediction tensors.
    eps : float
        The epsilon value to avoid division by zero.
    """
    def __init__(
        self,
        thresholds: List[float] = [16., 74., 133.],
        eps: float = 1e-4
        ) -> None:
        """
        Initialize the mCSI score.
        
        Parameters
        ----------
        thresholds : list of float, optional
            The thresholds to apply to the target and prediction tensors,
            by default [16., 74., 133.].
        eps : float, optional
            The epsilon value to avoid division by zero, by default 1e-4.
        """
        super().__init__()
        self.thresholds = thresholds
        self.eps = eps

    @staticmethod
    def _threshold(
        y_true: torch.FloatTensor,
        y_pred: torch.FloatTensor,
        threshold: float
        ) -> torch.FloatTensor:
        """
        Apply a threshold to both the target and the prediction tensors.

        Parameters
        ----------
        y_true : FloatTensor
            The target tensor.
        y_pred : FloatTensor
            The prediction tensor.
        threshold : float
            The threshold to apply.

        Returns
        -------
        FloatTensor
            The thresholded target tensor.
        FloatTensor
            The thresholded prediction tensor.
        """
        # Get the thresholded target and prediction tensors.
        y_true_res = (y_true >= threshold).float()
        y_pred_res = (y_pred >= threshold).float()

        # Set the NaN values of the target and prediction tensors to 0.
        is_nan = torch.isnan(y_true) | torch.isnan(y_pred)
        y_true_res[is_nan] = 0
        y_pred_res[is_nan] = 0

        return y_true_res, y_pred_res

    # TODO: uniformity in the order of arguments (y_pred, y_true)
    def forward(
        self,
        y_pred: torch.FloatTensor,
        y_true: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        Compute the mean Critical Success Index (CSI) score between the
        prediction and the target tensors. The CSI score is computed for
        different thresholds and then averaged.

        Parameters
        ----------
        y_pred : FloatTensor
            The prediction tensor of shape (B, T, H, W), where B is the batch
            size, T is the sequence length, H is the height and W is the
            width.
        y_true : FloatTensor
            The target tensor of shape (B, T, H, W), where B is the batch
            size, T is the sequence length, H is the height and W is the
            width.
        
        Returns
        -------
        FloatTensor
            The mean CSI score.
        """
        results = 0.

        with torch.no_grad():
            for thresh in self.thresholds:
                # Apply the threshold to the target and prediction tensors.
                y_true, y_pred = self._threshold(y_true, y_pred, thresh)
                # Get the number of hits, misses and false alarms.
                hits = torch.sum(y_true * y_pred, dim=(-2, -1)).int()
                misses = torch.sum(y_true * (1 - y_pred), dim=(-2, -1)).int()
                fas = torch.sum((1 - y_true) * y_pred, dim=(-2, -1)).int()
                # Compute the CSI score.
                csi = hits / (hits + misses + fas + self.eps)
                # Add the CSI score to the results.
                results += csi.mean()

        # Return the mean CSI score.
        return results / len(self.thresholds)
