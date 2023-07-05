from functools import partial
from torch.optim.optimizer import register_optimizer_step_post_hook
from udjat.watcher import Watcher

__version__ = "0.1.0"

__all__ = [
    "init",
]

def _optimizer_post_hook(optimizer, args, kwargs, watcher=None):
    watcher.step()

def init(
    **kwargs
):
    """
    This function creates a `Watcher` which handles user provided signals
    """
    watcher = Watcher(**kwargs)
    register_optimizer_step_post_hook(partial(_optimizer_post_hook, watcher=watcher))
    return watcher