# Trainy on-demand profiler

<p align="center">
  <img height='100px' src="https://www.ocf.berkeley.edu/~asai/static/images/trainy.png">
</p>

![GitHub Repo stars](https://img.shields.io/github/stars/Trainy-ai/trainy?style=social)
[![](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/TrainyAI)
[![](https://dcbadge.vercel.app/api/server/d67CMuKY5V)](https://discord.gg/d67CMuKY5V)

This is the trainy CLI and daemon to setup on demand tracing for PyTorch in pure Python. This will allow you to extract traces in the middle of training.

## Installation

You can either install from pypi or from source

```
# install from pypi
pip install trainy

# install from source
git clone https://github.com/Trainy-ai/trainy
pip install -e trainy
```

## Quickstart

If you haven't already, set up ray head and worker nodes. This can configured to happen automatically using [Skypilot](https://skypilot.readthedocs.io/en/latest/index.html) or K8s

```
# on the head node 
$ ray start --head --port 6380

# on the worker nodes
$ ray start --address ${HEAD_IP}
```

In your train code, initialize the trainy daemon before running your train loop.

```
import trainy
trainy.init()
Trainer.train()
```

While your model is training, to capture traces on all the nodes, run 

```
$ trainy trace --logdir ~/my-traces
```

This saves the traces for each process locally into `~/my-traces`. It's recommended
you run a shared file system like NFS or an s3 backed store so that all of your traces
are in the same place. An example of how to do this and scale this up is under the `examples/resnet_mnist`
on AWS 

## How It Works

Trainy registers a hook into whatever PyTorch optimizer is present in your code,
to count the optimizer iterations and registers the program with the head ray node. 
A separate HTTP server daemon thread is run concurrently, which waits for a trigger
POST request to start profiling.

## Need help 

We offer support for both setting up trainy and analyzing program traces. If you are interested,
please [email us](mailto:founders@trainy.ai)
