#!/bin/bash
#---Number of core
#SBATCH -c 1

#---Job's name in LSF system
#SBATCH -J example_job

#---Error file
#SBATCH -e example_job_err

#---Output file
#SBATCH -o example_job_out

#---Queue name
#SBATCH -q PQ_nbc

#---Partition
#SBATCH -p investor

##########################################################
# Setup envrionmental variable.
##########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=$NPROCS

. $MODULESHOME/../global/profile.modules
module load anaconda/2.7.13
module load fsl

##########################################################
##########################################################

python task-networks.py
