#!/bin/bash
wget -O 'RNN.pt' 'https://www.dropbox.com/s/lod4bldkysbln0w/RNN.pt?dl=1'
python3 RNN_test.py RNN.pt $1 $2 $3

