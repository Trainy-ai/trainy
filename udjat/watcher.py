import http.server
import json
import logging
import os
import socket
import socketserver

from collections import defaultdict
from warnings import warn
from torch.profiler import profile, schedule, tensorboard_trace_handler

from typing import Dict

if 'LOCAL_RANK' in os.environ:
    # Environment variables set by torch.distributed.launch or torchrun
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    MASTER_ADDR = os.environ['MASTER_ADDR']
else:
    LOCAL_RANK = 0
    WORLD_RANK = 0


## getting the hostname by socket.gethostname() method
hostname = socket.gethostname()
## getting the IP address using socket.gethostbyname() method
ip_address = socket.gethostbyname(hostname)

class Watcher:

    """
    class that handles external signals to initiate profile traces
    """

    _profile = None
    _num_new_steps = 0
    _current_step = -1
    _step_dict: Dict[str, int] = defaultdict(int)

    @classmethod
    def is_profiling(cls):
        return not cls._profile is None

    @classmethod
    def init_step_count(cls, requester: str):
        cls._step_dict[requester] = cls._current_step

    @classmethod
    def erase_step_count(cls, requester: str) -> bool:
        return cls._step_dict.pop(requester, None) is not None

    @classmethod
    def current_step(cls) -> int:
        return cls._current_step

    @classmethod
    def start(cls,
            wait=1,
            warmup=1,
            active=3,
            logdir='./log',
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ):
        if cls.is_profiling():
            logging.info("trace already in progress. Skipping this trace request")
            return
        config = {
            'schedule' : {
                'wait' : wait,
                'warmup' : warmup,
                'active' : active,
            },
            'profiler' : {
                'record_shapes' : record_shapes,
                'profile_memory' : profile_memory,
                'with_stack' : with_stack
            }
        }
        logdir = os.path.abspath(logdir)
        if LOCAL_RANK == 0: 
            logging.info(f"saving traces to {logdir}")
        cls._num_new_steps = wait + warmup + active
        cls._profile = profile(
            schedule=schedule(**config['schedule']),
            on_trace_ready=tensorboard_trace_handler(logdir),
            **config['profiler']
        )
        cls._profile.start()

    @classmethod
    def increment_step(cls, requester: str) -> int:
        """Increments the step count for the requester.
        returns global step count
        """
        if requester not in cls._step_dict:
            cls.init_step_count(requester)
        cls._step_dict[requester] += 1
        new_step = max(cls._step_dict.values())
        if new_step > cls._current_step:
            delta = new_step - cls._current_step
            if delta > 1:
                warn("Profiler step count has increased more than 1 - "
                     f"current_step = {cls._current_step} step dict =  {cls._step_dict}")
            for _ in range(0, delta):
                if cls.is_profiling(): cls._profile.step()
            cls._current_step = new_step
        if cls.is_profiling():
            if delta >= cls._num_new_steps:
                cls._profile.stop()
                cls._profile = None
            else:
                cls._num_new_steps -= delta
        return cls._current_step

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        profiler_config = json.loads(post_data.decode('utf-8'))
        Watcher.start(**profiler_config)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        reply = f'Starting trace on {ip_address}, WORLD_RANK = {WORLD_RANK}. Saving to {os.path.abspath(profiler_config["logdir"])}'
        logging.info(reply)
        self.wfile.write(reply.encode())

def start_server():
    server_address = ('0.0.0.0', 25000 + LOCAL_RANK)
    httpd = http.server.ThreadingHTTPServer(server_address, RequestHandler)
    httpd.serve_forever()
