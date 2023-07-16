import logging
import os
import ray
import subprocess
import threading
import time

from functools import partial
from torch.optim.optimizer import register_optimizer_step_post_hook
from udjat import constants
from udjat.httpd import start_server
from udjat.watcher import Watcher

import socket
## getting the hostname by socket.gethostname() method
hostname = socket.gethostname()
## getting the IP address using socket.gethostbyname() method
ip_address = socket.gethostbyname(hostname)

if 'LOCAL_RANK' in os.environ:
    # Environment variables set by torch.distributed.launch or torchrun
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    MASTER_ADDR = os.environ['MASTER_ADDR']
else:
    LOCAL_RANK = 0
    MASTER_ADDR = ip_address

__version__ = "0.1.0"

__all__ = [
    "init",
]

def _is_master_node():
    return ip_address == MASTER_ADDR

def _optimizer_post_hook(optimizer, args, kwargs):
    Watcher.increment_step("Optimizer")
    
def connect_ray():
    ray.init(address=f'ray://{MASTER_ADDR}:{constants.UDJAT_REMOTE_RAY_CLIENT_PORT}', ignore_reinit_error=True)

def init(
    **kwargs
):
    """
    Initialize `Watcher` which handles user provided signals
    """
    os.makedirs(constants.UDJAT_TMPDIR, exist_ok = True) 
    connect_ray()
    register_optimizer_step_post_hook(_optimizer_post_hook)
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.setDaemon(True)
    server_thread.start()