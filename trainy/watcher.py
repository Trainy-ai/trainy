import logging
import os

from collections import defaultdict
from torch.profiler import profile, schedule, tensorboard_trace_handler
from functools import partial
from ray.tune.utils.file_transfer import sync_dir_between_nodes
from warnings import warn
from typing import Dict

from trainy.constants import MASTER_ADDR, LOCAL_RANK, ip_address


def trace_handler(p, path):
    if path.startswith("s3"):
        raise NotImplementedError("s3 storage not implemented")
    tensorboard_trace_handler(path)(p)


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
    def start(
        cls,
        wait=1,
        warmup=1,
        active=3,
        logdir="./log",
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ):
        if cls.is_profiling():
            logging.info("trace already in progress. Skipping this trace request")
            return
        config = {
            "schedule": {
                "wait": wait,
                "warmup": warmup,
                "active": active,
            },
            "profiler": {
                "record_shapes": record_shapes,
                "profile_memory": profile_memory,
                "with_stack": with_stack,
            },
        }
        logdir = os.path.abspath(logdir)
        if LOCAL_RANK == 0:
            logging.info(f"saving traces to {logdir}")
        cls._num_new_steps = wait + warmup + active
        cls._profile = profile(
            schedule=schedule(**config["schedule"]),
            on_trace_ready=partial(trace_handler, path=logdir),
            **config["profiler"],
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
                warn(
                    "Profiler step count has increased more than 1 - "
                    f"current_step = {cls._current_step} step dict =  {cls._step_dict}"
                )
            for _ in range(0, delta):
                if cls.is_profiling():
                    cls._profile.step()
            cls._current_step = new_step
        if cls.is_profiling():
            if delta >= cls._num_new_steps:
                cls._profile.stop()
                cls._profile = None
            else:
                cls._num_new_steps -= delta
        return cls._current_step
