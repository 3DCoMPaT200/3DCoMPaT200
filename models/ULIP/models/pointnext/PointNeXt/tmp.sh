#!/bin/bash
#SBATCH --job-name=install_env
#SBATCH -N 1
#SBATCH -e /ibex/scratch/ahmems0a/install_env.err
#SBATCH -o /ibex/scratch/ahmems0a/install_env.out
#SBATCH --mail-user=mahmoud.ahmed@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=3:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:p100:1
module load cuda/11.3 
module load gcc/9.2.0 
source /home/ahmems0a/miniconda3/bin/activate
conda activate ulip
# module avail cuda
# bash install.sh
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
