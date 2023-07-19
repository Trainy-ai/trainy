import os
import ray
import threading
from datetime import datetime
from torch.optim.optimizer import register_optimizer_step_post_hook
from trainy.httpd import start_server
from trainy.watcher import Watcher
from trainy.constants import distribution_conf, hostname, MASTER_ADDR

from posthog import Posthog

os.environ["PYTHONHASHSEED"] = "0"

posthog = Posthog(
    project_api_key="phc_4UgX80BfVNmYRZ2o3dJLyRMGkv1CxBozPAcPnD29uP4",
    host="https://app.posthog.com",
)

__version__ = "0.1.3"

__all__ = [
    "init",
]


def _optimizer_post_hook(optimizer, args, kwargs):
    Watcher.increment_step("Optimizer")


def connect_ray():
    if not ray.is_initialized():
        ray.init(address="auto")


def init(**kwargs):
    """
    Initialize `Watcher` which handles user provided signals
    """
    connect_ray()
    posthog.capture(
        f"{hash(MASTER_ADDR if MASTER_ADDR != 'localhost' else hostname)}",
        event="initialized trainy daemon",
        properties=distribution_conf,
        timestamp=datetime.utcnow(),
    )
    register_optimizer_step_post_hook(_optimizer_post_hook)
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.setDaemon(True)
    server_thread.start()
