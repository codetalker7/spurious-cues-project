#!/bin/bash
#SBATCH --account=cs6966
#SBATCH --partition=coe-gpu-class-grn
#SBATCH --qos=coe-gpu-students-grn
#SBATCH --job-name=spurious_cues_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=logs/evals-%j.out
#SBATCH --error=logs/evals-%j.err

# load required modules
module load miniforge3
module load cuda/12.8.1
conda activate spurious_cues_project

# cd to the right dir
export SCRDIR=/scratch/general/vast/$USER/spurious_cues_project
cd $SCRDIR

# the logs directory
mkdir -p logs
mkdir -p evals

# running evals for all models
echo "Starting at $(date)"
python evaluation.py --model-name "allenai/Olmo-3-7B-Instruct" 
python evaluation.py --model-name "Qwen/Qwen2.5-1.5B-Instruct" 
python evaluation.py --model-name "Qwen/Qwen3-1.7B" 
python evaluation.py --model-name "Qwen/Qwen3-8B" 
echo "Ended at $(date)"

