# DLCV_FinalProject

## Required Package

* Numpy
* pytorch
* torchvision
* pickle
* PIL
* Scipy
* tqdm
* argparse

## Usage
Execute the following command to test the compact model for face recognition on the validation dataset and predict on the test dataset.
```
bash final.sh $1 $2 $3 $4

    $1: the compact model
    
        ┌─ depth_fire.pth
        └─ quantized_depth_fire.pth
        
    $2: path to the dataset of 2018-spring-dlcv-final-project-2
    
        $2
        ├─ test
        ├─ train
        ├─ train.txt
        ├─ val
        └─ val.txt
        
    
    $3: path to the test dataset
    
        $3
        ├─ 00001.jpg
        ├─ 00002.jpg
        │       .
        │       .
        │       .
        │       .
        ├─ 07151.jpg
        └─ 07152.jpg
    
    $4: output csv file of the prediction of test dataset
```

## Reference
1. "Squeeznet: Alexnet-level Accuracy with 50X fewer parameters and < 0.5 MB model size", Forrest N. landola et al., ICLR 2017
2. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", Andrew G. Howard et al., CoRR, abs/1704.04861, 2017
