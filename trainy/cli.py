import aiohttp
import asyncio
import click
import os
import ray
from datetime import datetime
from trainy import connect_ray, posthog
from trainy.constants import hostname

connect_ray()


async def post_profile(session, url, config):
    try:
        async with session.post(url, json=config) as resp:
            result = (
                await resp.text() if resp.status == 200 else f"request to {url} failed"
            )
            return result
    except aiohttp.client_exceptions.ClientConnectorError as e:
        return f"failed to trace {url}. Possibly due to no process running here"
    except aiohttp.client_exceptions.ServerDisconnectedError as e:
        return f"failed to trace {url}. Possibly due to no process running here"


async def trace_exec(config):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for node in ray.nodes():
            if node["Alive"]:
                num_gpus = (
                    int(node["Resources"]["GPU"]) if "GPU" in node["Resources"] else 0
                )
                for i in range(num_gpus):
                    url = f"http://{node['NodeManagerAddress']}:{25000 + i}"
                    tasks.append(
                        asyncio.ensure_future(post_profile(session, url, config))
                    )

        responses = await asyncio.gather(*tasks)
        for r in responses:
            print(r)


@click.group()
def trainy():
    pass


@trainy.command(help="Trigger a Pytorch Profiler trace in cluster")
@click.option("--logdir", default="./logs", help="path to save logs")
@click.option("--wait", default=2, help="num steps to wait before profiling")
@click.option("--warmup", default=2, help="num steps to warmup before profiling")
@click.option("--active", default=5, help="num steps to profile")
@click.option("--record_shapes", default=True, help="record input shapes")
@click.option("--profile_memory", default=True, help="profile memory footprint")
@click.option("--with_stack", default=True, help="include python stack traces")
def trace(logdir, wait, warmup, active, record_shapes, profile_memory, with_stack):
    config = {
        "wait": wait,
        "warmup": warmup,
        "active": active,
        "record_shapes": record_shapes,
        "profile_memory": profile_memory,
        "with_stack": with_stack,
        "logdir": os.path.abspath(logdir),
    }
    posthog.capture(
        f"{hash(hostname)}",
        event="trace requested",
        properties=config,
        timestamp=datetime.utcnow(),
    )
    asyncio.run(trace_exec(config))
