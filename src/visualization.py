from typing import List, Tuple
import matplotlib.pyplot as plt

def plot_metrics(
    train_mse_history: List[Tuple[int, float]],
    train_val_mse_history: List[Tuple[int, float]],
    **kwargs
    ) -> None:
    """
    Plot the training and validation MSE history along with other
    possible validation metrics.
    
    Parameters
    ----------
    train_mse_history : List[Tuple[int, float]]
        The training MSE history.
    train_val_mse_history : List[Tuple[int, float]]
        The validation MSE history.
    kwargs
        Additional validation metrics.
    """
    # Create a new figure.
    plt.figure(figsize=(15, 5))
    # Plot the training MSE.
    plt.plot(*zip(*train_mse_history), label='Train MSE')
    # Plot the validation MSE.
    plt.plot(*zip(*train_val_mse_history), label='Validation MSE')
    # Set the x label.
    plt.xlabel('Epoch')
    # Set the y label.
    plt.ylabel('MSE')
    # Set the legend.
    plt.legend()
    # Show the plot.
    plt.show()
    # Loop over the additional metrics.
    for metric, history in kwargs.items():
        # Create a new figure.
        plt.figure(figsize=(15, 5))
        # Plot the metric.
        plt.plot(*zip(*history), label=metric)
        # Set the x label.
        plt.xlabel('Epoch')
        # Set the y label.
        plt.ylabel(metric)
        # Set the legend.
        plt.legend()
        # Show the plot.
        plt.show()
