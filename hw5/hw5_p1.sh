#!/bin/bash
wget -O 'FC_for_C.pt' 'https://www.dropbox.com/s/oqlkp4tn4ltms4n/FC_for_C.pt?dl=1'
python3 ResNet_test.py FC_for_C.pt $1 $2 $3

