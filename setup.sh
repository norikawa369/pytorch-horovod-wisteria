#!/bin/bash

#PJM -g ge58
#PJM -m e
#PJM -L rscgrp=short-a
#PJM -L node=1

HOME=/work/ge58/e58004

module load cuda/12.1
module load nccl/2.17.1
module load gcc/8.3.1
module load ompi/4.1.1

python3 -m venv .venv
source .venv/bin/activate

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install cmake
# horovodはtorch=2.1.0と非互換性あり、以下のように明示的にC++17でやるとinstallできる
HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=$NCCL_HOME pip install --no-cache-dir git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17
pip install pandas==2.0.0
pip install rdkit==2022.9.5
pip install scikit-learn==1.2.2
pip install matplotlib==3.7.1
pip install typed-argument-parser==1.8.0
pip install tape-proteins==0.5
pip install lifelines==0.27.4
pip install hydra-core

deactivate