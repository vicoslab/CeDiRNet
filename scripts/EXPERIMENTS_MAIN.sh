#!/bin/sh

function wait_or_interrupt() {
  # set to kill any child processes if parent is interupted
  child_ids=$(pgrep -P $$ | xargs echo | tr " " ,)
  trap "pkill -P $child_ids && pkill $$" SIGINT
  # now wait
  if [ -z "$1" ] ; then
    wait
  elif [ -n "$1" ] && [ -n "$2" ] ; then
    MAX_CAPACITY=$1
    INDEX=$2
    # wait only if INDEX mod MAX_CAPACITY == 0
    if [ $((INDEX % MAX_CAPACITY)) -eq 0 ] ; then
      wait
    fi
  else
    # wait if more child processes exist than allowed ($1 is the number of allowed children)
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
  fi
}


export DISABLE_X11=1

###################################################
######## LEARNING
###################################################

export SERVERS="localhost:0,1,2,3,4,5,6,7"
NUM_TRAINING_GPUS=8

for db in "sorghum" "carpk" "pucpr+"; do
  export DATASET=$db
  for hardneg in 0 16; do
    if [[ "$hardneg" == 16 ]] && [[ "$db" == "sorghum" ]] ; then
      hardneg=64
    fi
    for weight_decay in 0 1e-04; do
      echo "DATASET=$db hardneg=$hardneg weight_decay=$weight_decay"
      ./run_distributed.sh train.py --config model.weight_decay=$weight_decay \
                                 train_dataset.hard_samples_size=$hardneg \
                                 center_model.kwargs.dilated_nn_args.freeze_learning=True \
                                 center_model.lr=0 \
                                 loss_w.w_cent=0 \
                                 num_gpus=$NUM_TRAINING_GPUS \
                                 display=False save_interval=5 skip_if_exists=True&
      wait_or_interrupt
    done
  done
done

for db in "acacia_06" "acacia_12" "oilpalm"; do
  export DATASET=$db
  for hardneg in 0 16; do
    for weight_decay in 0 1e-04; do
      for fold in 0 1 2; do
        echo "DATASET=$db hardneg=$hardneg weight_decay=$weight_decay"
        ./run_distributed.sh train.py --config model.weight_decay=$weight_decay \
                                   train_dataset.kwargs.fold=$fold \
                                   train_dataset.hard_samples_size=$hardneg \
                                   center_model.kwargs.dilated_nn_args.freeze_learning=True \
                                   center_model.lr=0 \
                                   loss_w.w_cent=0 \
                                   display=False save_interval=5 skip_if_exists=True &
        wait_or_interrupt
      done
    done
  done
done

wait_or_interrupt

###################################################
######## TESTING/INFERENCE
###################################################

GPU_LIST=("localhost:0" "localhost:1" "localhost:2" "localhost:3")
GPU_COUNT=${#GPU_LIST[@]}

s=0
for db_type in "test" "val"; do
  for db in "sorghum" "carpk" "pucpr+" "acacia_06" "acacia_12" "oilpalm"; do
    if [[ "$db" == sorghum ]] ; then
       ALL_EPOCH=("" _005 _010 _015 _020 _025 _030 _035 _040)
    else
       ALL_EPOCH=("" _100 _105 _150 _155 _005 _010 _015 _020 _025 _030 _035 _040 _050 _055 _060 _065 _070 _075 _080 _085 _090 _095 _110 _115 _120 _125 _130 _135 _140 _145 _160 _165 _170 _175 _180 _185 _190 _195)
    fi
    export DATASET=$db
    for hardneg in 0 16; do
      if [[ "$hardneg" == 16 ]] && [[ "$db" == sorghum ]] ; then
        hardneg=64
      fi
      for weight_decay in 0 1e-04; do
        for epoch_eval in "${ALL_EPOCH[@]}"; do
          # run with generic, pre-trained localization network
          SERVERS=${GPU_LIST[$((s % GPU_COUNT))]} ./run_distributed.sh test.py --config eval_epoch=$epoch_eval \
                                     train_settings.train_dataset.polar_gt_opts.center_ignore_px=$center_ignore_px \
                                     train_settings.train_dataset.hard_samples_size=$hardneg \
                                     train_settings.model.weight_decay=$weight_decay \
                                     train_settings.num_gpus=$NUM_TRAINING_GPUS \
                                     center_checkpoint_name_list=None \
                                     dataset.kwargs.type=$db_type \
                                     display=False skip_if_exists=True &
          s=$((s+1))
          wait_or_interrupt $GPU_COUNT
          # run with hand-crafted (1D conv) localization network
          SERVERS=${GPU_LIST[$((s % GPU_COUNT))]} ./run_distributed.sh test.py --config eval_epoch=$epoch_eval \
                                     center_checkpoint_name="handcrafted_localization" \
                                     center_checkpoint_path=None \
                                     center_model.kwargs.use_learnable_nn=False \
                                     train_settings.train_dataset.hard_samples_size=$hardneg \
                                     train_settings.model.weight_decay=$weight_decay \
                                     train_settings.num_gpus=$NUM_TRAINING_GPUS \
                                     dataset.kwargs.type=$db_type \
                                     display=False skip_if_exists=True &
            s=$((s+1))
            wait_or_interrupt $GPU_COUNT
        done
      done
    done
  done
done

wait_or_interrupt