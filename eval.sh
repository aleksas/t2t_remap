#!/bin/bash
PROBLEM=num_to_text
MODEL=transformer
WORKER_GPU=2
HPARAMS_SET=transformer_base_bs94_lrc1_do4_f

USR_DIR=.
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET

DECODE_FILE=$DATA_DIR/decode_this.txt
DECODE_TO_FILE=$DATA_DIR/decode_result.txt
echo "Einam namo." > $DECODE_FILE
echo "Nerandu namo." >> $DECODE_FILE
echo "Reikia kelių kartų, kad atsinaujintų populiacija." >> $DECODE_FILE
echo "Stuktelėjo keletą kartų." >> $DECODE_FILE

BEAM_SIZE=5
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USR_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$DECODE_TO_FILE
