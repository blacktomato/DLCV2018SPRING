#!/bin/bash
wget -O 'FCN_8s.h5' 'https://www.dropbox.com/s/7w6vxo79mix9zrl/FCN_8s.h5?dl=1'
python3 ./FCN_test.py ./FCN_8s.h5 $1 $2
