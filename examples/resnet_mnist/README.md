This is an example run command for this example

```
torchrun  \
        --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=172.31.7.183 --master_port=1234\
        resnet_main.py \
        --backend=nccl --batch_size=256 --arch=resnet18
```