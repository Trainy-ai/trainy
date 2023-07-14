import logging
import os
import ray
import subprocess
import threading

from functools import partial
from torch.optim.optimizer import register_optimizer_step_post_hook
from udjat import constants
from udjat.watcher import Watcher, start_server

if 'LOCAL_RANK' in os.environ:
    # Environment variables set by torch.distributed.launch or torchrun
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    MASTER_ADDR = os.environ['MASTER_ADDR']

__version__ = "0.1.0"

__all__ = [
    "init",
]

def _is_master_node():
    import socket
    ## getting the hostname by socket.gethostname() method
    hostname = socket.gethostname()
    ## getting the IP address using socket.gethostbyname() method
    ip_address = socket.gethostbyname(hostname)
    return ip_address == MASTER_ADDR or MASTER_ADDR in ['localhost', '127.0.0.1']

def _optimizer_post_hook(optimizer, args, kwargs):
    Watcher.increment_step("Optimizer")

def init(
    **kwargs
):
    """
    Initialize `Watcher` which handles user provided signals
    """
    import time
    logging.info("attaching to udjat ray daemon")
    try:
        if _is_master_node() and LOCAL_RANK == 0:
            subprocess.run(f'ray start --head --port {constants.UDJAT_REMOTE_RAY_PORT} --ray-client-server-port {constants.UDJAT_REMOTE_RAY_CLIENT_PORT}', 
            shell=True,
            stderr=subprocess.DEVNULL)
        elif not _is_master_node() and LOCAL_RANK == 0:
            subprocess.run(f"ray start --address='{MASTER_ADDR}:{constants.UDJAT_REMOTE_RAY_PORT}'",
            shell=True,
            stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logging.info(f'ray is already started on {WORLD_RANK}')
    
    MAX_RETRIES = 5
    retries = 0
    while not ray.is_initialized() and retries < MAX_RETRIES:
        try:
            ray.init(address=f'ray://{MASTER_ADDR}:{constants.UDJAT_REMOTE_RAY_CLIENT_PORT}')
        except:
            print(f'ray head not created yet on {MASTER_ADDR}. Trying again in 5 seconds')
            time.sleep(5)
        retries += 1
    
    register_optimizer_step_post_hook(_optimizer_post_hook)
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.setDaemon(True)
    server_thread.start()