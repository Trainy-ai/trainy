This is an example run command for this example on a single node

```
torchrun  \
        --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=172.31.7.183 --master_port=1234\
        resnet_main.py \
        --backend=nccl --batch_size=256 --arch=resnet18
```

You can scale this up the number of nodes running and onto the cloud (AWS) using the following command. Refer to [this guide to install skypilot](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)

```
sky launch -c mycluster skypilot_resnet.yaml
```

To test out the profiler, login to the head node and trigger tracing.

```
ssh mycluster
trainy trace
```
