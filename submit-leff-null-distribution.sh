#!/bin/bash
#---Number of core
#SBATCH -c 1

#---Job's name in LSF system
#SBATCH -J idconn_retrieval_leff-null_job

#---Error file
#SBATCH -e idconn_retrieval_leff-null_job_err

#---Output file
#SBATCH -o idconn_retrieval_leff-null_job_out

#---Queue name
#SBATCH --qos pq_nbc

#---Partition
#SBATCH -p investor

#---Account
#SBATCH --account acc_nbc


##########################################################
# Setup envrionmental variable.
##########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=$NPROCS

. $MODULESHOME/../global/profile.modules

module load anaconda/2.7.13

##########################################################
##########################################################

srun python /home/kbott006/physics-retrieval/idconn-retrieval/idconn-retrieval/network-analysis/leff-null-distribution.py
