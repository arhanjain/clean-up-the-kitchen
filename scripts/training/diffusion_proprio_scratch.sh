#!/bin/bash

# Check if a command-line argument is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/dataset training_run_name"
    exit 1
fi

DATASET_PATH=$1
TRAINING_RUN_NAME=$2

# Check if the provided path is a file
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: '$DATASET_PATH' is not a valid file."
    exit 1
fi


echo "Splitting $DATASET_PATH into train and validation sets inplace..."
python robomimic/robomimic/scripts/split_train_val.py --dataset $DATASET_PATH

echo "Training a Diffusion model from scratch on $DATASET_PATH"
python train.py \
  --config ./scripts/training/configs/diffusion_proprio_config.json \
  --name $TRAINING_RUN_NAME \
  --dataset $DATASET_PATH \
