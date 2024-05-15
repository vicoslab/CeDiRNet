#!/usr/bin/env bash

###################################################
######## ENVIRONMENT SETTINGS (CONDA, ENVs, ...)
###################################################


ROOT_DIR=~/cedirnet/

CONDA_HOME=~/conda
CONDA_ENV_NAME=CeDiRNet

echo "Loading conda env ..."

. $CONDA_HOME/etc/profile.d/conda.sh

conda activate $CONDA_ENV_NAME
echo "... done - using $CONDA_ENV_NAME"

export SOURCE_DIR=$ROOT_DIR/src
export OUTPUT_DIR=$ROOT_DIR/exp

export SORGHUM_DIR=$ROOT_DIR/datasets/SorghumPlantCenters2016
export CARPK_DIR=$ROOT_DIR/datasets/CARPK
export TREE_COUNTING_DIR=$ROOT_DIR/datasets/tree_counting_dataset/croped_dataset_512x512

###################################################
######## DATA PARALLEL SETTINGS
###################################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_SHM_DISABLE=1

###################################################
######## RUN TASK
###################################################

cd $SOURCE_DIR
python $CMD_ARGS