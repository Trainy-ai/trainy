from functools import partial
from torch.optim.optimizer import register_optimizer_step_post_hook
from udjat.watcher import Watcher

__version__ = "0.1.0"

__all__ = [
    "init",
]

def _optimizer_post_hook(optimizer, args, kwargs):
    Watcher.increment_step("Optimizer")

def init(
    **kwargs
):
    """
    Initialize `Watcher` which handles user provided signals
    """
    register_optimizer_step_post_hook(_optimizer_post_hook)
    return Watcher