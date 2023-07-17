import os
import socket

## getting the hostname by socket.gethostname() method
hostname = socket.gethostname()
## getting the IP address using socket.gethostbyname() method
ip_address = socket.gethostbyname(hostname)

if "LOCAL_RANK" in os.environ:
    # Environment variables set by torch.distributed.launch or torchrun
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    WORLD_RANK = int(os.environ["RANK"])
    MASTER_ADDR = os.environ["MASTER_ADDR"]
else:
    LOCAL_RANK = 0
    LOCAL_WORLD_SIZE = 1
    WORLD_SIZE = 1
    WORLD_RANK = 0
    MASTER_ADDR = ip_address


