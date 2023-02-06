#!/bin/bash
#
# name of job
#SBATCH --job-name=MNIST-VAE-training
# set the number of processors/tasks needed
#SBATCH -n 8
#SBATCH -N 1
#
# account to use
#SBATCH -q scavenger
#SBATCH --account=pi_nurit.haspel
#
# set max wallclock time  DD-HH:MM:SS
#SBATCH --time=02-00:00:00
#
# Set filenames for stdout and stderr.  %j jobid, %x job name
#SBATCH --error=AECompare/chimera_jobs/%x-%j.err
#SBATCH --output=AECompare/chimera_jobs/%x-%j.out
#
#SBATCH --partition=DGXA100
#SBATCH --mem=4gb
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=fatemeh.afrasiabi001@umb.edu
#
echo `date`
for i in 30 35 40 45 50 55 60 65 70 75 80 90 100
do
    python train.py --model 'VAE' --latent_len $i --random_seed 35354 
    python train.py --model 'VAE' --latent_len $i --random_seed 35354 --epochs 50
    python train.py --model 'VAE' --latent_len $i --random_seed 35354 --learning_rate 0.0001
    python train.py --model 'VAE' --latent_len $i --random_seed 35354 --learning_rate 0.0001 --epochs 50
done
# Diagnostic/Logging Information
echo "end time is `date`"
