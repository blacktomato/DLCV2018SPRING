#!/bin/bash
wget -O 'RNN_S2S.pt' 'https://www.dropbox.com/s/98nxu58wgqkozmj/RNN_S2S.pt?dl=1'
wget -O 'CNN_S2S.pt' 'https://www.dropbox.com/s/xeegdj2yijo7lpm/CNN_S2S.pt?dl=1'
python3 RNN_S2S_test.py RNN_S2S.pt CNN_S2S.pt $1 $2

