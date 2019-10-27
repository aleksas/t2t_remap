#!/bin/bash
PROBLEM=num_to_text
MODEL=transformer
HPARAMS_SET=transformer_base_bs94_lrc1_do4_f
HPARAMS_SET=transformer_base_multistep12_bs94_lrws10

USR_DIR=.
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET

#curl -o ngrok.zip https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-386.zip
#unzip ngrok.zip
#nohup ./ngrok http 8888 &
#sleep 3
#curl http://127.0.0.1:4040/api/tunnels

tensorflow_model_server \
  --port=8888 \
  --model_name=$MODEL \
  --model_base_path=$TRAIN_DIR/export
