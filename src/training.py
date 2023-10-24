from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

from fastprogress import progress_bar
import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .checkpoint import Checkpoint
from .dataloader import NoisifyDataloader, ValidationDataloader
from .metrics import mCSI
from .model import UNet2D, UNet2DTemporalCondition, UNet3D
from .wandb_utils import log_images, save_model
from .scaler import Scaler

class MiniTrainer:
    """
    Class to train a model for the diffusion process.
    
    Attributes
    ----------
    train_dataloader : NoisifyDataloader
        The dataloader for the training set.
    valid_dataloader : NoisifyDataloader
        The dataloader for the validation set.
    loss_func : Module
        The loss function to use.
    psnr : PeakSignalNoiseRatio
        The PSNR metric.
    ssim : StructuralSimilarityIndexMeasure
        The SSIM metric.
    mcis : mCSI
        The mCSI metric.
    model : UNet2D
        The model to train.
    scaler : GradScaler
        The gradient scaler to use for training.
    device : str
        The device to use for training.
    sampler : (model: UNet2D, past_frames: FloatTensor) -> FloatTensor
        The sampler to use for the diffusion process.
    val_batch : Tuple
        The validation batch to use for logging and validation.
    n_predicted : int
        The number of frames to predict.
    optimizer : Optimizer
        The optimizer to use for training.
    scheduler : Scaler
        The scheduler to use for training.
        
    Methods
    -------
    train_step : (loss: FloatTensor) -> None
        Performs a training step on the model based on the loss.
    one_epoch : (epoch: int, optional) -> None
        Performs one epoch of training.
    one_epoch_validate : (epoch: int, optional) -> None
        Performs one epoch of validation.
    """
    def __init__(
        self,
        train_dataloader: NoisifyDataloader,
        valid_dataloader: ValidationDataloader,
        model: Union[UNet2D, UNet2DTemporalCondition, UNet3D],
        sampler: Callable[[UNet2D, torch.FloatTensor], torch.FloatTensor],
        ir069_scaler: Scaler,
        device: str = 'cuda',
        loss_func: nn.Module = nn.MSELoss(),
        n_frames_to_predict: int = 1,
        # TODO: change to auto_regression_steps 
        n_auto_regression_steps: Union[int, None] = 3,
        channels_per_image: int = 1,
        gradient_clipping: bool = False,
        ) -> None:
        """
        Initializes the MiniTrainer class.

        Parameters
        ----------
        train_dataloader : NoisifyDataloader
            The dataloader for the training set.
        valid_dataloader : ValidationDataloader
            The dataloader for the validation set.
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
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_func = loss_func
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.m_csi = mCSI(ir069_scaler=ir069_scaler).to(device)
        self.model = model.to(device)
        self.scaler = torch.cuda.amp.GradScaler()
        self.ir069_scaler = ir069_scaler
        self.device = device
        self.sampler = sampler
        self.val_batch = next(iter(valid_dataloader))
        self.n_auto_regression_steps = n_auto_regression_steps
        self.n_frames_to_predict = n_frames_to_predict
        self.channels_per_image = channels_per_image
        self.checkpoint = Checkpoint()
        self.gradient_clipping = gradient_clipping

    def train_step(self, loss: torch.FloatTensor) -> None:
        """
        Performs a training step.

        Parameters
        ----------
        loss : FloatTensor
            The loss to backpropagate.
        """
        # Zero the gradients.
        self.optimizer.zero_grad()
        # Apply the scale factor to the loss.
        self.scaler.scale(loss).backward()
        # Apply the step on the optimizer.
        self.scaler.step(self.optimizer)
        # Update the scale factor.
        self.scaler.update()
        # Appply the step on the 1 cycle learning rate scheduler.
        self.scheduler.step()

    def one_epoch(
        self,
        epoch: int,
        train_mse_history: List[Tuple[int, float]]
        ) -> None:
        """
        Train one epoch, log the metrics and save the model.

        Parameters
        ----------
        epoch : int, optional
            The epoch number, by default None.
        """
        # Set the model to train mode.
        self.model.train()
        # Initialize the progress bar with the training dataloader.
        pbar = progress_bar(self.train_dataloader, leave=True)
        # Loop over the batches.
        for batch in pbar:
            # Get the frames, the temperature and the noise.
            frames, t, noise = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)  # __to_device gives weird error!
            b, _, _, h, w = frames.shape
            # Reshape the frames and the noise to match the model input shape.
            if type(self.model) == UNet2D or type(self.model) == UNet2DTemporalCondition:
                frames = frames.reshape(b, -1, h, w)
                noise = noise.reshape(b, -1, h, w)
            #self.__to_device(batch, device=self.device)
            # Squeeze the noise on the second dimension.
            # noise = noise.squeeze(1)
            # Apply mixed precision on the prediction.
            with torch.autocast('cuda'):
                # Predict the noise through the model.
                predicted_noise = self.model(frames, t)
                # Compute the loss on the actual noise and the predicted noise.
                loss = self.loss_func(noise, predicted_noise)
            # Apply the training step using the loss.
            self.train_step(loss)
            # Clip the gradients if gradient clipping is enabled.
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            # Log on wandb the loss and the learning rate.
            wandb.log({
                'train_mse': loss.item(),
                'learning_rate': self.scheduler.get_last_lr()[0]
                })
            # Add the loss score to the progress bar.
            pbar.comment = f'train epoch={epoch}, MSE={loss.item():2.3f}'
            
            train_mse_history.append((epoch, loss.item()))

    def one_epoch_validation(
        self,
        epoch: int,
        config: SimpleNamespace,
        val_mse_history: List[Tuple[int, float]],
        val_psnr_history: List[Tuple[int, float]],
        val_ssim_history: List[Tuple[int, float]],
        val_m_csi_history: List[Tuple[int, float]],
        ) -> None:
        # Validation is always done in the first validation loader batch for time purposes.
        past_frames, val_target_frames = self.val_batch[0].to(self.device), self.val_batch[1].to(self.device)
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
                past_frames = past_frames[:,1:]
            # Get the predicted frames by cutting the last n_predicted frames.
            predictions = past_frames[:,-self.n_auto_regression_steps:]

        # Compute the metrics on the predicted frames and the target frames.
        psnr_metric = self.psnr(predictions[:, :, 0], val_target_frames[:,:,0]).float().cpu()
        ssim_metric = self.ssim(predictions[:, :, 0], val_target_frames[:,:,0]).float().cpu()
        mse_metric = self.loss_func(predictions[:, :, 0], val_target_frames[:,:,0]).float().cpu()
        m_csi_metric = self.m_csi(predictions[:, :, 0], val_target_frames[:,:,0]).float().cpu()

        # Print the metrics.
        print(f'validation epoch={epoch},' +\
            f' val PSNR={psnr_metric.item():2.3f},' +\
            f' val SSIM={ssim_metric.item():2.3f},' +\
            f' val MSE={mse_metric.item():2.3f},' +\
            f' val mCSI={m_csi_metric.item():2.3f}')

        # Log the metrics on wandb.
        wandb.log({
            'val_psnr': psnr_metric,
            'val_ssim': ssim_metric,
            'val_mse': mse_metric,
            'val_m_csi': m_csi_metric})

        # Save the metrics for plotting purposes.
        val_mse_history.append((epoch, mse_metric.item()))
        val_psnr_history.append((epoch, psnr_metric.item()))
        val_ssim_history.append((epoch, ssim_metric.item()))
        val_m_csi_history.append((epoch, m_csi_metric.item()))

        # Plot the predicted and target frames and log them on wandb.
        log_images(
            val_target_frames[0, :, 0],
            predictions[0, :, 0],
            scaling_values=(-.5, -5))
        
        # Save the model parameters locally based on the mse metric.
        self.checkpoint.save_best(
            self.model,
            config.model_name,
            val_mse=mse_metric.item(),
            val_m_csi=m_csi_metric.item(),
            val_psnr=psnr_metric.item(),
            val_ssim=ssim_metric.item(),
            )

    def __prepare(self, config: SimpleNamespace) -> None:
        """
        Define the total train steps based on the number of epochs and the
        length of the train dataloader. Moreover, initialize the optimizer and
        the learning rate scheduler.

        Parameters
        ----------
        config : SimpleNamespace
            The configuration of the pipeline.
        """
        # Update the config of wandb.
        wandb.config.update(config)
        # Set the total train steps of the config.
        config.total_train_steps = config.epochs * len(self.train_dataloader)
        # Set the optimizer and the scheduler.
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=config.lr, total_steps=config.total_train_steps)

    def fit(
        self,
        config: SimpleNamespace
        ) -> Tuple[
            List[Tuple[int, float]],
            List[Tuple[int, float]],
            List[Tuple[int, float]],
            List[Tuple[int, float]],
            List[Tuple[int, float]]]:
        # Prepare the pipeline.
        self.__prepare(config)
        # Get the validation past frames and target frames.
        # Validation is always done in the first validation loader batch for time purposes.
        #val_past_frames, val_target_frames = self.val_batch[0].to(self.device), self.val_batch[1].to(self.device)

        # Create array of metrics for plotting purposes.
        train_mse_history = []
        val_mse_history = []
        val_psnr_history = []
        val_ssim_history = []
        val_m_csi_history = []

        # Loop over the epochs.
        for epoch in progress_bar(range(config.epochs), total=config.epochs, leave=True):
            # Apply the training step on one epoch.
            self.one_epoch(epoch, train_mse_history)

            # If the validation is enabled, apply the validation step.
            #if config.validate_epochs:
            #    self.one_epoch_validation(epoch)

            # Log the model predictions on wandb on the validation set.
            if epoch % config.log_every_epoch == 0:
                self.one_epoch_validation(
                    epoch,
                    config,
                    val_mse_history,
                    val_psnr_history,
                    val_ssim_history,
                    val_m_csi_history)

        # Save the best model according to checkpointing on wandb.
        save_model(config.model_name)

        # Return the metrics history.
        histories = (
            train_mse_history,
            val_mse_history,
            val_psnr_history,
            val_ssim_history,
            val_m_csi_history)

        return histories

    def __to_device(
        t: Union[List[float], Tuple[float, ...], torch.FloatTensor],
        device: str = 'cpu'
        ) -> Union[List[float], Tuple[float, ...], torch.FloatTensor]:
        """
        Get an iterable of tensors or a tensor and move it to the device.

        Parameters
        ----------
        t : list of float | tuple of float | FloatTensor
            The iterable of tensors or the tensor to move to the device.
        device : str, optional
            The device to move the tensors to, by default 'cpu'.

        Returns
        -------
        list of float | tuple of float | FloatTensor
            The iterable of tensors or the tensor moved to the device.
        """
        if isinstance(t, (tuple, list)):
            return [_t.to(device) for _t in t]
        elif isinstance(t, torch.Tensor):
            return t.to(device)
        else:
            raise("Not a Tensor or list of Tensors")

'''def parse_args(config):
    "A brute force way to parse arguments, it is probably not a good idea to use it"
    parser = argparse.ArgumentParser(description='Run training baseline')
    for k,v in config.__dict__.items():
        parser.add_argument('--'+k, type=type(v), default=v)
    args = vars(parser.parse_argsf())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)''';