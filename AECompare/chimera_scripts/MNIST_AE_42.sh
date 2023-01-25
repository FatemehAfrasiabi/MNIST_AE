#!/bin/bash
#
# name of job
#SBATCH --job-name=MNIST-AE-training
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
for i in 10 12 15 16 18 20 21 22 23 24 25 26 27 28 29 30
do
    python train.py --latent_len $i --random_seed 4352
done
# Diagnostic/Logging Information
echo "end time is `date`"
