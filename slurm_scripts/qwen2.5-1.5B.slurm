#!/bin/bash
#SBATCH --account=cs6966
#SBATCH --partition=coe-gpu-class-grn
#SBATCH --qos=coe-gpu-students-grn
#SBATCH --job-name=spurious_cues_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qwen2.5-1.5B_generations-%j.out
#SBATCH --error=logs/qwen2.5-1.5B_generations-%j.err

# clear any loaded modules
module purge

# load required modules
module load miniforge3
module load cuda/12.8.1
conda activate spurious_cues_project

# cd to the right dir
export SCRDIR=/scratch/general/vast/$USER/spurious_cues_project
cd $SCRDIR

# the logs directory
mkdir -p logs
mkdir -p experiments

# monitoring the gpu
/uufs/chpc.utah.edu/sys/installdir/chpcscripts/gpu/gpu-monitor.sh &

# start the execution
echo "Starting evaluation at $(date)"

# python main2.py \
#     --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
#     --bbh-tasks navigate \
#     --verbose \

python analyse_logit_lens.py \
    --experiment-file "experiments/20260405-022401-navigate-Qwen2.5-1.5B-Instruct-biastypeans_always_a-thinkingFalse.json"

# logit lens analysis (existing)
python analyse_logit_lens.py --experiment-file "experiments/20260405-022401-navigate-Qwen2.5-1.5B-Instruct-biastypeans_always_a-thinkingFalse.json"

# new: probing + cosine similarity + DLA
python analyse_representations.py --experiment-file "experiments/20260405-022401-navigate-Qwen2.5-1.5B-Instruct-biastypeans_always_a-thinkingFalse.json"

echo "Evaluation finished at $(date)"
