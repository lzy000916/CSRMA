#!/bin/sh
echo "Start Train"
dir=$(dirname "$0")
if [ -f "$dir/Train.py" ];then
    cd $dir
    pwd
    export CUDA_VISIBLE_DEVICES=0
    ##### run with Polyp Datasets
    #python Train.py --model_name CSRMAUNet --epoch 120 --batchsize 8 --trainsize 352 --train_save CSRMAUNet_Kvasir_1e4_bs8_e120_s352_25 --lr 0.0001 --train_path $dir/data/TrainDataset --test_path $dir/data/TestDataset/Kvasir/
    #sleep 1m
    python Test.py --train_save CSRMAUNet_Kvasir_1e4_bs8_e120_s352_25 --testsize 352 --test_path $dir/data/TestDataset
    
  
else
    echo "file not exists"
fi
