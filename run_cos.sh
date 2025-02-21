echo "Start Train"
dir=$(dirname "$0")
if [ -f "$dir/Train_cos.py" ];then
    cd $dir
    pwd
    export CUDA_VISIBLE_DEVICES=0
    ##### run with Polyp Datasets
    python Train_cos.py --model_name CSRMAUNet --epoch 50 --batchsize 4 --trainsize 352 --train_save CSRMAUNet_Kvasir_1e4_bs16_e360_s352 --lr 0.0001 --train_path $dir/data/TrainDataset --test_path $dir/data/TestDataset/Kvasir/
    sleep 1m
    python Test.py --train_save CSRMAUNet_Kvasir_1e4_bs16_e360_s352 --testsize 352 --test_path $dir/data/TestDataset
    
  
else
    echo "file not exists"
fi
