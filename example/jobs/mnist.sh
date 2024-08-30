#!/bin/bash

#PJM -g ge58
#PJM -m e
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM --mpi proc=8

HOME=/work/ge58/e58004

module load cuda/12.1
module load nccl/2.17.1
module load gcc/8.3.1
module load ompi/4.1.1

GPUS_PER_NODE=`nvidia-smi -L | wc -l`

cd ../../
source .venv/bin/activate
cd example/

echo "num of mpi process: ${PJM_MPI_PROC}"
echo "gpus per node: ${GPUS_PER_NODE}"

mpirun -np ${PJM_MPI_PROC} -machinefile ${PJM_O_NODEINF} \
    -map-by ppr:${GPUS_PER_NODE}:node -mca pml ob1 \
    -mca btl_tcp_if_include ib0,ib1,ib2,ib3 \
    python mnist.py \
        out_dir='./result/mnist/' \
        training.log_interval=50 \
        training.batch_size=64 \
        training.batch_size_test=64 \


deactivate