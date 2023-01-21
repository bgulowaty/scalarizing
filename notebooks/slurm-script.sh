#!/bin/bash
#SBATCH -N1
#SBATCH -c8
#SBATCH --mem=16gb
#SBATCH --time=6:00:00
#SBATCH -D /home/bogul/scalarizing/notebooks

cd "$(dirname "$0")"


eval $@ 
