#!/bin/bash
#SBATCH --account=cs6966
#SBATCH --partition=coe-gpu-class-grn
#SBATCH --qos=coe-gpu-students-grn
#SBATCH --job-name=spurious_cues_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --output=logs/llm_as_judge-%j.out
#SBATCH --error=logs/llm_as_judge-%j.err

# load required modules
module load miniforge3
module load cuda/12.8.1
conda activate spurious_cues_project

# cd to the right dir
export SCRDIR=/scratch/general/vast/$USER/spurious_cues_project
cd $SCRDIR

# set groq api key
export GROQ_API_KEY=

# start the execution
echo "Starting evaluation at $(date)"

python llm_as_judge.py --model-name Qwen/Qwen3-8B

echo "Evaluation finished at $(date)"

