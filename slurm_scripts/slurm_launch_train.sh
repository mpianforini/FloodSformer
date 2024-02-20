#!/bin/sh

#SBATCH --job-name=FloodSformer_train
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100_80g:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12GB
#SBATCH --time=06:00:00
#SBATCH --open-mode=append

echo "JOB NAME    : $SLURM_JOB_NAME"
echo "JOB ID      : $SLURM_JOB_ID"
echo "PARTITION   : $SLURM_JOB_PARTITION"
echo "MEMORY      : $SLURM_MEM_PER_NODE MB"
echo "HOSTNAME    : $HOSTNAME"
echo "GPU ID      : $SLURM_JOB_GPUS"

module purge
module load miniconda3    # CHANGE
source "$CONDA_PREFIX/etc/profile.d/conda.sh"  # CHANGE
conda activate floodsformer  # CHANGE: name of the conda virtual environment

# CHANGE: path to the current working directory
WORKINGDIR=/hpc/scratch/name.surname/FloodSformer

echo "WORKINGDIR  : ${WORKINGDIR}"
echo '====================='
echo

cd ${WORKINGDIR}
python3 ${WORKINGDIR}/run_net.py \
        --cfg ${WORKINGDIR}/configs/Parflood/DB_Parma_20m_trainVPTR.yaml  # CHANGE: use the path to the file .yaml with the appropriate parameters (see README.md)

conda deactivate