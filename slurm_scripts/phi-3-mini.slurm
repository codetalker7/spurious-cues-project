#!/bin/bash
#SBATCH --account=cs6966
#SBATCH --partition=coe-gpu-class-grn
#SBATCH --qos=coe-gpu-students-grn
#SBATCH --job-name=spurious_cues_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/Phi-3-mini-4k-%j.out
#SBATCH --error=logs/Phi-3-mini-4k-%j.err

# clear any loaded modules
# module purge

# load required modules
# module load miniforge3
module load cuda/12.4.0


# make sure conda works in batch jobs
export PATH=/scratch/general/vast/$USER/miniconda3/bin:$PATH
source /scratch/general/vast/$USER/miniconda3/etc/profile.d/conda.sh

# activate environment using full path
conda activate /scratch/general/vast/$USER/miniconda3/envs/spurious_cues_project

python -c "import torch; import numpy; print(torch.__version__, numpy.__version__)"
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

python main.py \
    --model-name "microsoft/Phi-3-mini-4k-instruct" \
    --bbh-tasks causal_judgment date_understanding disambiguation_qa hyperbaton logical_deduction_five_objects movie_recommendation navigate ruin_names snarks sports_understanding temporal_sequences tracking_shuffled_objects_three_objects web_of_lies \
    --verbose 
echo "Evaluation finished at $(date)"
