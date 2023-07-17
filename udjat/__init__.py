import ray
import threading
from torch.optim.optimizer import register_optimizer_step_post_hook
from udjat.httpd import start_server
from udjat.watcher import Watcher

__version__ = "0.1.0"

__all__ = [
    "init",
]


def _optimizer_post_hook(optimizer, args, kwargs):
    Watcher.increment_step("Optimizer")


def connect_ray():
    if not ray.is_initialized():
        ray.init(address='auto')


def init(**kwargs):
    """
    Initialize `Watcher` which handles user provided signals
    """
    connect_ray()
    register_optimizer_step_post_hook(_optimizer_post_hook)
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.setDaemon(True)
    server_thread.start()
