from types import SimpleNamespace
from typing import Callable, List, Tuple, Union

from fastprogress import progress_bar
import torch
from torch import nn
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import wandb

from .dataloader import ValidationDataloader
from .metrics import mCSI
from .model import UNet2D, UNet2DTemporalCondition, UNet3D
from .wandb_utils import log_images, save_model
from .scaler import Scaler

class Evaluator:

    def __init__(
        self,
        test_dataloader: ValidationDataloader,
        model: Union[UNet2D, UNet2DTemporalCondition, UNet3D],
        sampler: Callable[[UNet2D, torch.FloatTensor], torch.FloatTensor],
        ir069_scaler: Scaler,
        event_type: str,
        device: str = 'cuda',
        n_frames_to_predict: int = 1,
        n_auto_regression_steps: Union[int, None] = 3,
        channels_per_image: int = 1,
        ) -> None:
        """
        Initializes the MiniTrainer class.

        Parameters
        ----------
        valid_dataloader : ValidationDataloader
            The dataloader for the test set.
        model : UNet2D
            The model to train.
        sampler : (model: UNet2D, past_frames: FloatTensor) -> FloatTensor
            The sampler function to use for the diffusion process.
        device : str, optional
            The device to use for training, by default 'cuda'.
        loss_func : Module, optional
            The loss function to use, by default MSELoss
        n_predicted : int, optional
            The number of frames to predict, by default 3.
        """
        self.test_dataloader = test_dataloader
        self.mse: nn.Module = nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.m_csi = mCSI(ir069_scaler=ir069_scaler).to(device)
        self.model = model.to(device)
        self.scaler = torch.cuda.amp.GradScaler()
        self.ir069_scaler = ir069_scaler
        self.device = device
        self.sampler = sampler
        self.n_auto_regression_steps = n_auto_regression_steps
        self.n_frames_to_predict = n_frames_to_predict
        self.channels_per_image = channels_per_image
        self.event_type = event_type

    def evaluate(
        self,
        ) -> Tuple[float, float, float, float]:

        pbar = progress_bar(self.test_dataloader, leave=True)
        # Loop over the batches.

        running_psnr = 0.
        running_ssim = 0.
        running_mse = 0.
        running_m_csi = 0.

        for i, batch in enumerate(pbar):
            past_frames, test_target_frames = batch[0].to(self.device), batch[1].to(self.device)
            b, _, c, h, w = past_frames.shape
  
            if self.n_auto_regression_steps is None or self.n_auto_regression_steps == 0:
                prediction_frames = self.sampler(self.model, past_frames=past_frames).to(self.device)
                if type(self.model) == UNet2D or type(self.model) == UNet2DTemporalCondition:
                    predictions = prediction_frames.reshape(
                        b,
                        self.n_frames_to_predict,
                        c,
                        h,
                        w)
                else:
                    predictions = prediction_frames
            else:
                # Prepare the validation progress bar.
                pbar = progress_bar(range(self.n_auto_regression_steps), leave=True)
                # Apply autoregressive sampling.
                for _ in pbar:
                    # Predict the next frame.
                    prediction_frames = self.sampler(self.model, past_frames=past_frames).to(self.device)
                    # Reshape the predictions to their original size.
                    if type(self.model) == UNet2D or type(self.model) == UNet2DTemporalCondition:
                        prediction_frames = prediction_frames.reshape(
                            b,
                            self.n_frames_to_predict,
                            c,
                            h,
                            w)
                    # Add the prediction to the past frames.
                    past_frames = torch.cat([past_frames, prediction_frames], dim=1)
                    # Remove the first frame from the past frames.
                    past_frames = past_frames[:, 1:]
                # Get the predicted frames by cutting the last n_predicted frames.
                predictions = past_frames[:,-self.n_auto_regression_steps:]

            # Compute the metrics on the predicted frames and the target frames.
            psnr_metric = self.psnr(predictions[:, :, 0], test_target_frames[:, :, 0]).float().cpu().item()
            ssim_metric = self.ssim(predictions[:, :, 0], test_target_frames[:, :, 0]).float().cpu().item()
            mse_metric = self.mse(predictions[:, :, 0], test_target_frames[:, :, 0]).float().cpu().item()
            m_csi_metric = self.m_csi(predictions[:, :, 0], test_target_frames[:, :, 0]).float().cpu().item()

            running_psnr += psnr_metric
            running_ssim += ssim_metric
            running_mse += mse_metric
            running_m_csi += m_csi_metric

            # Print the metrics.
            print(f'test step for {self.event_type}={i + 1}/{len(self.test_dataloader)},' +\
                f' test PSNR={running_psnr / (i + 1):2.3f},' +\
                f' test SSIM={running_ssim / (i + 1):2.3f},' +\
                f' test MSE={running_mse / (i + 1):2.3f},' +\
                f' test mCSI={running_m_csi / (i + 1):2.3f}')

            # Plot the predicted and target frames of the first batch and log them on wandb.
            if i == 0:
                log_images(
                    test_target_frames[0, :, 0],
                    predictions[0, :, 0],
                    scaling_values=(-.5, -5),
                    log_name=f'Sample {self.event_type}')

        return (
          running_mse / len(self.test_dataloader),
          running_psnr / len(self.test_dataloader),
          running_ssim/ len(self.test_dataloader),
          running_m_csi/ len(self.test_dataloader))
