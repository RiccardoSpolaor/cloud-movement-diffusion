from typing import Any, Dict, List, Tuple
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

def plot_model_parameters_vs_m_csi(
    models_dictionaries: Dict[str, Any]
    ) -> None:
    """
    Plot the model parameters number for the different models
    along with the validation mCSI.

    Parameters
    ----------
    models_dictionaries : { str: Any }
        The models dictionaries.
    """
    plt.figure(figsize=(8, 5))
    for model, d in models_dictionaries.items():
        #label = model.removeprefix('cloud-diffuser-')
        #label = ' '.join([w.capitalize() for w in label.split('-')])
        plt.scatter(
            d['model_parameters'],
            d['val_m_csi'],
            s=90,
            marker='D',
            label=model,
            edgecolors='black'
            )
    plt.legend()
    plt.title('Validation mCSI with respect to Model Parameters Number')
    plt.ylabel('Valdidation mCSI')
    plt.tight_layout()
    plt.xlabel('Model Parameters Number')
    plt.margins(.2, .2)
    plt.xscale('log')
    plt.show()
    
def plot_metric(
    metric: str,
    metric_label: str,
    models_dictionaries: Dict[str, Any]
    ) -> None:
    """
    Plot the validation metric for the different models.
    
    Parameters
    ----------
    metric : str
        The metric to plot.
    metric_label : str
        The metric label.
    models_dictionaries : { str: Any }
        The models dictionaries.
    """
    plt.figure(figsize=(8, 5))
    labels = []
    for i, (model, d) in enumerate(models_dictionaries.items()):
        #label = model.removeprefix('cloud-diffuser-')
        #label = ' '.join([w.capitalize() for w in label.split('-')])
        labels.append(model)
        plt.bar(
            i,
            height=d[metric],
            width=.8,
            edgecolor='black',
            )
    plt.title(f'Validation {metric_label} comparison for the Different Models')
    plt.ylabel(f'Validation {metric_label}')
    plt.tight_layout()
    plt.xlabel('Model Name')
    plt.xticks(
        range(len(models_dictionaries)),
        labels,
        rotation=45,
        ha='right')
    if metric == 'val_m_csi':
        plt.ylim(.6, .9)
    if metric == 'val_ss_ssim':
        plt.ylim(.4, .9)
    plt.show()

def plot_metric_on_test_set(
    metric: str,
    metric_label: str,
    events_dictionaries: Dict[str, Any],
    title: str = None
    ) -> None:
    """
    Plot the test metric for the different events.
    
    Parameters
    ----------
    metric : str
        The metric to plot.
    metric_label : str
        The metric label.
    events_dictionaries : { str: Any }
        The events dictionaries.
    """
    if title is None:
        title = f'Test {metric_label} comparison for the Different Events'
    plt.figure(figsize=(8, 5))
    labels = []
    for i, (event, d) in enumerate(events_dictionaries.items()):
        labels.append(event)
        plt.bar(
            i,
            height=d[metric],
            width=.8,
            edgecolor='black')
    plt.title(title)
    plt.ylabel(f'Test {metric_label}')
    plt.tight_layout()
    plt.xlabel('Event Kind')
    plt.xticks(
        range(len(events_dictionaries)),
        labels,
        rotation=45,
        ha='right')
    plt.show()