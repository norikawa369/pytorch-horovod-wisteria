#!/bin/bash

#PJM -g ge58
#PJM -m e
#PJM -L rscgrp=short-a
#PJM -L node=2
#PJM --mpi proc=16

HOME=/work/ge58/e58004

module load cuda/12.1
module load nccl/2.17.1
module load gcc/8.3.1
module load ompi/4.1.1

GPUS_PER_NODE=`nvidia-smi -L | wc -l`

cd ../
source .venv/bin/activate

echo "num of mpi process: ${PJM_MPI_PROC}"
echo "gpus per node: ${GPUS_PER_NODE}"

mpirun -np ${PJM_MPI_PROC} -machinefile ${PJM_O_NODEINF} \
    -map-by ppr:${GPUS_PER_NODE}:node -mca pml ob1 \
    -mca btl_tcp_if_include ib0,ib1,ib2,ib3 \
    python check_gpus.py  

deactivate