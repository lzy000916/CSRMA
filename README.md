# CSRMA-Net
This repository is the official implementation of the paper Channel and space reconstruction mixed attention in polyp  segmentation with CSRMA-Net

🚀 **This project is currently under active development. Please stay tuned for updates!**

## Datasets

Polyp Datasets (include Kvasir-SEG, CVC-ClinicDB,  ETIS, and CVC-300.) \[[From PraNet](https://github.com/DengPingFan/PraNet)\]:
Total: \[[Aliyun](http://118.31.22.118:5244/Aliyun/CSCAUNet/Datasets/Polyp%205%20Datasets.zip)\], \[[Baidu]( https://pan.baidu.com/s/1q5I2e2bbwXdW4evJdCAUpg?pwd=1111)\]
TrainDataset: \[[Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)\] 
TestDataset: \[[Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)\]


## Usage
### Create Environment

First of all, you need to have a `pytorch` environment, I use `pytorch 1.10`, but it should be possible to use a lower version, so you can decide for yourself.

You can also create a virtual environment using the following command (note: this virtual environment is named `pytorch`, if you already have a virtual environment with this name on your system, you will need to change `environment.yml` manually).

```shell
conda env create -f docs/enviroment.yml
```

### Training

You can run the following command directly:

```shell
sh run.sh ### use stepLR
sh run_cos.sh ### use CosineAnnealingLR 
```

If you only want to run a single dataset, you can comment out the irrelevant parts of the `sh` file, or just type something like the following command from the command line:

```shell
python Train.py --model_name CSCAUNet --epoch 121 --batchsize 16 --trainsize 352 --train_save CSCAUNet_Kvasir_1e4_bs16_e120_s352 --lr 0.0001 --train_path $dir/data/TrainDataset --test_path $dir/data/TestDataset/Kvasir/  # you need replace ur truely Datapath to $dir.
```

### Get all predicted results (.png)

If you use a `sh` file for training, it will be tested after the training is complete.

If you use the `python` command for training, you can also comment out the training part of the `sh` file, or just type something like the following command at the command line:

```shell
python Test.py --train_save CSCAUNet_Kvasir_1e4_bs16_e120_s352 --testsize 352 --test_path $dir/data/TestDataset
```

### Evaluating

- For evaluating the polyp dataset, you can use the `matlab` code in `eval` or use the evaluation code provided by \[[UACANet](https://github.com/plemeri/UACANet)\].
- For other datasets, you can use the code in [evaldata](https://github.com/z872845991/evaldata/).
- The reason for using a different evaluation code is to use the same methodology in the evaluation as other papers that did experiments on the dataset.


## Acknowledgement

