#!/bin/bash                                                                                                                  
#SBATCH --account=project_462000824                                                                                          
#SBATCH --output=./log/out_%j                                                                                     
#SBATCH --error=./log/err_%j                                                                                      
#SBATCH --partition=dev-g                                                                                 
#SBATCH --ntasks-per-node=1                                                                                                  
#SBATCH --cpus-per-task=7                                                                                                    
#SBATCH --gpus-per-node=8                                                                                           
#SBATCH --mem=320G                                                                                                  
#SBATCH --time=01:00:00

export OMP_NUM_THREADS=1 

module purge                                                                                                                 
module use /appl/local/csc/modulefiles/                                                                                      
module load pytorch/2.7                                                                                                      
                                                                                                                             
export SING_IMAGE=/pfs/lustrep4/appl/local/csc/soft/ai/images/pytorch_2.7.1_lumi_vllm-0.8.5.post1.sif                        
export HF_HOME=/scratch/project_462000824/cache/hf/hub
export HF_HUB_ENABLE_HF_TRANSFER=1

source /scratch/project_462000824/amoisala/venv/bin/activate

# Run script
srun torchrun --standalone --nnodes=1 --nproc_per_node=8 ingest.py
