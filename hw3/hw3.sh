#!/bin/bash
wget -O 'FCN_32s.h5' 'https://www.dropbox.com/s/xopvlxr7ujuqu90/FCH_32s.h5?dl=1'
python3 ./FCN_test.py ./FCN_32s.h5 $1 $2

