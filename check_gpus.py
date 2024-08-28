import torch
import horovod
import horovod.torch as hvd
import os

# Horovodの初期化
hvd.init()

# プロセスのランクやサイズを取得
rank = hvd.rank()
size = hvd.size()
local_size = hvd.local_size()

print(f"Rank: {rank}, Size: {size}, Local Size: {local_size}")