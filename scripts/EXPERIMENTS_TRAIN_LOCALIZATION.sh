#!/bin/bash

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

export DISABLE_X11=0
export DATASET=synt-center-learn-weakly

export SERVERS="localhostA:0,1,2,3,4,5,6,7 localhostB:0,1,2,3,4,5,6,7"

###################################################
######## Run learning with l1 loss
###################################################

for w_cent in 1; do
  ./run_distributed.sh train.py --config loss_w.w_cent=$w_cent \
                            train_dataset.batch_size=128 \
                            train_dataset.hard_samples_size=64 \
                            train_dataset.kwargs.length=500 \
                            num_gpus=2 \
                            display=True &
  wait_or_interrupt
done


###################################################
######## Run learning with focal loss
###################################################

LOSS_STR=('{"type":"focal","args":{"alpha":0.75,"gamma":5,"delta":4,"A":0.02}}'
          '{"type":"focal","args":{"alpha":0.75,"gamma":2,"delta":4,"A":0.0625}}')
LOSS_NAME=('focal_delta=4-A=0.02-gamma=5'
           'focal_delta=4-A=0.0625-gamma=2')

LOSS_COUNT=${#LOSS_NAME[@]}

for loss_id in $(eval echo "{0..$((LOSS_COUNT-1))}") ; do
  for w_cent in 10 1; do
    ./run_distributed.sh train.py --config  center_model.lr=0.01 \
                                loss_w.w_cent=$w_cent \
                                "loss_name=${LOSS_NAME[loss_id]}" \
                                "loss_opts.localization_loss=${LOSS_STR[loss_id]}" \
                                display=False &
    wait_or_interrupt
  done
done

wait_or_interrupt