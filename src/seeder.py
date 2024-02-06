import random
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility. The seed is set for the 
    random library, the numpy library and the pytorch library.
    
    Parameters
    ----------
    seed : int, optional
        The random seed to use for reproducibility, by default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# TODO: Shall we keep the function below ir the one above
'''def set_seed(seed: int = 42, is_reproducible: bool = False) -> None:
    """
    Set the random seed for the random, torch, and numpy libraries.

    Parameters
    ----------
    seed : int
        The seed to use for the random, torch, and numpy libraries.
    is_reproducible : bool, optional
        Whether to demand reproducibility from the torch library,
        by default False.
    """
    try: torch.manual_seed(seed)
    except NameError: pass
    try: torch.cuda.manual_seed_all(seed)
    except NameError: pass
    try: np.random.seed(seed % (2**32-1))
    except NameError: pass
    random.seed(seed)
    if is_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False''';
