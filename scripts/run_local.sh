#!/usr/bin/env bash

###################################################
######## ENVIRONMENT SETTINGS (CONDA, ENVs, ...)
###################################################

echo "Loading conda env ..."

. /home/domen/conda/etc/profile.d/conda.sh

conda activate CeDiRNet
echo "... done - using CeDiRNet"

STORAGE_DIR=/storage/

export SOURCE_DIR=$STORAGE_DIR/user/Projects/DIVID/CeDiRNet/src
export OUTPUT_DIR=$STORAGE_DIR/user/Projects/center-vector-exp

export SORGHUM_DIR=$STORAGE_DIR/datasets/SorghumPlantCenters2016
export CARPK_DIR=$STORAGE_DIR/datasets/CARPK
export TREE_COUNTING_DIR=$STORAGE_DIR/datasets/tree_counting_dataset/croped_dataset_512x512
export KOLEKTOR_COUNT_DIR=$STORAGE_DIR/private/kolektor/object_detection_dataset

###################################################
######## DATA PARALLEL SETTINGS
###################################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2

###################################################
######## RUN TASK
###################################################

cd $SOURCE_DIR
python $CMD_ARGS