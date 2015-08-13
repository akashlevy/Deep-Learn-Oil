#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-60:00
#SBATCH --gre=gpu:1
#SBATCH -p holyseasgpu
#SBATCH --mem=1000
#SBATCH -o hostname.out
#SBATCH -e hostname.err
#SBATCH --mail-type=END
#SBATCH --mail-user=YOUR EMAIL

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python fcn.py
