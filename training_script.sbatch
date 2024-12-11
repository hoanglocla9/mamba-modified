#!/bin/bash
#SBATCH -p hgx2q # partition (queue) #hgx2q
#SBATCH -N 1 # number of nodes
#SBATCH -n 8  # number of cores
#SBATCH --job-name=train_mamba
#SBATCH --gres=gpu:2
#SBATCH -t 7-00:00 # time (D-HH:MM)
#SBATCH -o logs/slurm.%N.%j.out # STDOUT
#SBATCH -e logs/slurm.%N.%j.err # STDERR

module load slurm

echo "Job started at:" `date +"%Y-%m-%d %H:%M:%S"`

echo "LAUNCH EXPERIMENTS"
#(cd /home/lhloc249/MISO_NAS/ && python3 train_nas.py -c 4 -p 8085 --strategy "random" -n 200 -t "ref_ch4(ppm)" -m "MAExLatency")
#(cd /home/lhloc249/Projects/mamba-modified && python train.py)
export CUDA_VISIBLE_DEVICES=6,7
#conda /home/lhloc249/D1/anaconda3/envs/research_dev/bin/activate
cd /home/lhloc249/Projects/mamba-modified
#python -m torch.distributed.launch train.py
torchrun --nproc_per_node 2 train.py 


echo "Job ended at:" `date +"%Y-%m-%d %H:%M:%S"`
