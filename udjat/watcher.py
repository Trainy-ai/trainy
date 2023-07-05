import torch
import logging

class Watcher:

    """
    class that handles external signals to initiate profile traces
    """

    _profile = None
    _step_count = 0
    _total_count = 0

    @property
    def is_profiling(cls):
        return not cls._profile is None

    def start(cls,
            wait=1,
            warmup=1,
            active=3,
            logdir='./log',
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ):
        if cls.is_profiling:
            logging.info("trace already in progress")
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
        cls._total_count = wait + warmup + active
        cls._profile = torch.profiler.profile(
            schedule=torch.profiler.schedule(**config['schedule']),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
            **config['profiler']
        )

    def step(cls):
        if not cls.is_profiling:
            return
        
        cls._profile.step()
        cls._step_count += 1
        
        if cls._step_count >= cls._total_count:
            cls._profile.stop()
            cls._profile = None
            cls._step_count = 0