# Gaussian noise training

#!/bin/bash
#
#SBATCH -J RASPING-NOISE-GAUSSIAN
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH --reservation=1g.10gb
#SBATCH --array=0-179

# Define arrays
noiseParams=("0.01" "0.03" "0.05" "0.07" "0.09" "0.11")
model_names=("reverse" "hist" "sort" "most-freq" "shuffle_dyck1" "shuffle_dyck2")
iterations=(1 2 3 4 5)

# Get the current array task ID
array_index=${SLURM_ARRAY_TASK_ID}

# Calculate param and itteration indices
param_index=$(((array_index % 30) / 5))
model_index=$((array_index / 30))
iteration_index=$((array_index % 5))

# Set the parameters based on the calculated indices
export NOISE_PARAM="${noiseParams[param_index]}"
export MODEL_NAME="${model_names[model_index]}"
export SEED="${iterations[iteration_index]}"
export TRAIN_LOSS_FILE_NAME="${MODEL_NAME}_train_loss_gaussian_std${NOISE_PARAM}_${SEED}"
export TRAIN_ACC_FILE_NAME="${MODEL_NAME}_train_acc_gaussian_std${NOISE_PARAM}_${SEED}"
export VAL_LOSS_FILE_NAME="${MODEL_NAME}_val_loss_gaussian_std${NOISE_PARAM}_${SEED}"
export VAL_ACC_FILE_NAME="${MODEL_NAME}_val_acc_gaussian_std${NOISE_PARAM}_${SEED}"

apptainer exec --nv ../container.sif python runOvertraining.py -baseModel $MODEL_NAME -maxLength 10 -n_epochs 100000 -seed $SEED -noiseType gaussian -noiseParam $NOISE_PARAM -saveDirectory savedData/noiseTrainingGaussian/ -trainLossFileName $TRAIN_LOSS_FILE_NAME -trainAccFileName $TRAIN_ACC_FILE_NAME -valLossFileName $VAL_LOSS_FILE_NAME -valAccFileName $VAL_ACC_FILE_NAME


# Regular overtraining script
#!/bin/bash
#
#SBATCH -J RASPING-OVERTRAIN
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH --reservation=1g.10gb
#SBATCH --array=0-35

# Define arrays
version=(1 2 3)
model_names=("reverse" "hist" "sort" "most-freq" "shuffle_dyck1" "shuffle_dyck2")
doRandom=("False" "True")

# Get the current array task ID
array_index=${SLURM_ARRAY_TASK_ID}

# Calculate indices
version_index=$(((array_index % 6) / 2))
model_index=$((array_index / 6))
random_index=$((array_index % 2))

# Set the parameters based on the calculated indices
export MODEL_NAME="${model_names[model_index]}"
export RANDOM_WEIGHTS="${doRandom[random_index]}"
export MAX_LENGTH="$((version[version_index]*5))"
if [ "$RANDOM_WEIGHTS" == "True" ]; then
    export TRAIN_LOSS_FILE_NAME="random_train_${MODEL_NAME}_loss_v${version[version_index]}"
    export TRAIN_ACC_FILE_NAME="random_train_${MODEL_NAME}_acc_v${version[version_index]}"
    export VAL_LOSS_FILE_NAME="random_val_${MODEL_NAME}_loss_v${version[version_index]}"
    export VAL_ACC_FILE_NAME="random_val_${MODEL_NAME}_acc_v${version[version_index]}"
else
    export TRAIN_LOSS_FILE_NAME="train_${MODEL_NAME}_loss_v${version[version_index]}"
    export TRAIN_ACC_FILE_NAME="train_${MODEL_NAME}_acc_v${version[version_index]}"
    export VAL_LOSS_FILE_NAME="val_${MODEL_NAME}_loss_v${version[version_index]}"
    export VAL_ACC_FILE_NAME="val_${MODEL_NAME}_acc_v${version[version_index]}"
fi

apptainer exec --nv ../container.sif python runOvertraining.py -baseModel $MODEL_NAME -maxLength $MAX_LENGTH -randomWeights $RANDOM_WEIGHTS -n_epochs 50000 -saveDirectory savedData/overTrainingV2/ -trainLossFileName $TRAIN_LOSS_FILE_NAME -trainAccFileName $TRAIN_ACC_FILE_NAME -valLossFileName $VAL_LOSS_FILE_NAME -valAccFileName $VAL_ACC_FILE_NAME
