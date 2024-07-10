#!/bin/bash
#SBATCH -c 1 # Number of cores requested
#SBATCH -t 30 # Runtime in minutes
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -p gpu_requeue
#SBATCH --mem=40000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o jax_optimization_holes-%j.out # Standard out goes to this file
#SBATCH -e jax_optimization_holes-%j.err # Standard err goes to this filehostname
#SBATCH --mail-user=joeri.lenaerts36@gmail.com

# load modules
module load python/3.10.13-fasrc01

# load python packages
mamba activate optimization

# memory overload
export TF_GPU_ALLOCATOR=cuda_malloc_async

# run code
python3 'jax_optimization_holes.py' phase_matching worst_case --epochs 1000 --verbose

mamba deactivate

